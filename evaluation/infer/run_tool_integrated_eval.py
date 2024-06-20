import argparse
import os
import sys
import time
from pdb import set_trace as bp
import_path = os.path.abspath(__file__)
for _ in range(2):
    import_path = os.path.dirname(import_path)
sys.path.append(import_path)
import copy
from collections import Counter
import pandas as pd

from tqdm import tqdm
import regex
import json
import random
from copy import deepcopy
from functools import partial
from vllm import LLM, SamplingParams
from pebble import ProcessPool
from transformers import AutoTokenizer
from concurrent.futures import TimeoutError

from eval.utils import generate_completions, load_hf_lm_and_tokenizer
from eval.python_executor import PythonExecutor
from eval.eval_script import *
from data_processing.answer_extraction import *
from data_processing.process_utils import *
from few_shot_prompts import *

from run_subset_parallel import markup_question

def extract_code(text):
    if not text.strip().endswith("```"):
        return ""
    if text.startswith("```python"):
        text = "hey\n" + text
    blocks = [block.split("```", 1)[0].strip() for block in text.split("```python") if '```' in block]
    blocks = [block for block in blocks if block]
    if not blocks:
        return ""
    code = []
    for block in blocks[:-1]:
        for line in block.split("\n"):
            if line.startswith("    ") or line.startswith("import") or line.startswith("def "):
                code.append(line)
            elif 'print(' not in line:
                code.append(line)
    code = "\n".join(code) + "\n" + blocks[-1]
    return code.strip()

def finish_answer_prediction(text):
    patt = regex.search(r"\\boxed{(?P<ans>.+)}", text)
    return patt is not None and patt.group('ans').strip()

def evaluate(eval_fn, tasks, _timeout=15, modular=None):
    with ProcessPool() as pool:
        timeout_cnt = 0
        iterator = pool.map(eval_fn, tasks, timeout=_timeout).result()
        labels = []
        while True:
            try:
                labels.append(int(next(iterator)))
                if isinstance(modular, int):
                    labels[-1] = labels[-1] % modular
            except StopIteration:
                break
            except TimeoutError as error:
                labels.append(0)
                timeout_cnt += 1
            except Exception as error:
                print(error.traceback, flush=True)
                exit()
    return labels, timeout_cnt


def infer(sample, n_repetition=1):
    test_data = [copy.deepcopy(sample) for _ in range(n_repetition)]
    if PROMPT_FORMAT == 'few_shot':
        assert FEW_SHOT_PROMPT is not None
        prompting = eval(FEW_SHOT_PROMPT)()

    prompts = []
    for example in test_data:
        prompt = ""
        if PROMPT_FORMAT == 'few_shot':
            prompt = prompting.format_prompt(example['messages'][-2]['content'], example['messages'][-1]['content'])
        else:
            for mess in example['messages']:
                if PROMPT_FORMAT == 'sft':
                    if mess['role'] == 'user':
                        prompt += f"User: {mess['content'].strip()}\n\nAssistant:"
                    elif mess['role'] == 'assistant':
                        prompt += mess['content'].strip()
                else:
                    raise NotImplementedError()
            prompt = prompt.lstrip()
        example['prompt'] = prompt
        prompts.append(prompt.lstrip())
    model_outputs = [item['messages'][-1]['content'].strip() for item in test_data]
    unfinished_ids = list(range(len(prompts)))

    executor = PythonExecutor(get_answer_from_stdout=True)

    n_iters = 2
    global model, tokenizer
    pbar = tqdm()
    while n_iters and unfinished_ids:
        pbar.update(1)
        model_inputs = [prompts[i] for i in unfinished_ids]
        finish_completion = None
        print("Loading model and tokenizer...")
        if USE_VLLM:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME_OR_PATH, trust_remote_code=True)
                print(f"{'-' * 20} prompt_to_ids {'-' * 20}\n{tokenizer.encode(model_inputs[0])}\n{'-' * 50}", flush=True)
                print(f"eos_token: {tokenizer.eos_token}", flush=True)
            if model is None:
                model = LLM(model=MODEL_NAME_OR_PATH, dtype='half', kv_cache_dtype="auto", max_model_len=2048, swap_space=4, gpu_memory_utilization=0.99, enforce_eager=True, tokenizer=TOKENIZER_NAME_OR_PATH, trust_remote_code=True, tensor_parallel_size=1)
                # model = LLM(model=args.model_name_or_path, dtype='half', kv_cache_dtype="fp8_e4m3", max_model_len=2048, swap_space=4, gpu_memory_utilization=0.4, enforce_eager=True, tokenizer=args.tokenizer_name_or_path, trust_remote_code=True, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
                # model = LLM(model=args.model_name_or_path, dtype='half', max_model_len=2048, swap_space=4, gpu_memory_utilization=0.4, enforce_eager=True, tokenizer=args.tokenizer_name_or_path, trust_remote_code=True, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
            stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
            if not NO_EXECUTION:
                stop_words.append("```output")
            if PROMPT_FORMAT == 'few_shot':
                stop_words.extend(prompting.stop_words())
            outputs = model.generate(model_inputs, SamplingParams(temperature=TEMPERATURE, top_p=1.0, max_tokens=1024, n=1, stop=stop_words))
            outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
            # finish_completion = [output.outputs[0].token_ids[-1] == tokenizer.eos_token_id for output in outputs]
            finish_completion = [output.outputs[0].token_ids[-1] == tokenizer.eos_token_id if (len(output.outputs) > 0 and len(output.outputs[0].token_ids) > 0) else False for output in outputs]
            outputs = [output.outputs[0].text for output in outputs]
        else:
            if model is None or tokenizer is None:
                model, tokenizer = load_hf_lm_and_tokenizer(
                    model_name_or_path=MODEL_NAME_OR_PATH, 
                    tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH, 
                    load_in_8bit=LOAD_IN_8BIT, 
                    load_in_half=LOAD_IN_HALF,
                    gptq_model=GPTQ
                )

            stop_id_sequences = [tokenizer.encode("```output", add_special_tokens=False)]
            if tokenizer.eos_token_id is not None:
                stop_id_sequences.append([tokenizer.eos_token_id])
            outputs, finish_completion = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=model_inputs,
                max_new_tokens=512,
                batch_size=EVAL_BATCH_SIZE,
                stop_id_sequences=stop_id_sequences,
                end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
            )

        if len(unfinished_ids) != len(outputs):
            print(f"input-output mismatch >>> {len(unfinished_ids)} != {len(outputs)}", flush=True)
            print(f"----- DEBUG -----\ninputs:\n{model_inputs[:10]}\noutputs:\n{str(outputs[:10])}\n----- DEBUG -----\n", flush=True)
            raise RuntimeError()

        if finish_completion is None:
            finish_completion = [finish_answer_prediction(output) for output in outputs]

        print("extract code ...", flush=True)
        codes = []
        code_indices = []
        for i, output, is_finished in zip(unfinished_ids, outputs, finish_completion):
            output = output.rstrip()
            if not NO_EXECUTION and not is_finished:
                code = extract_code(model_outputs[i] + output)
                if code:
                    codes.append(code)
                    code_indices.append(i)
            prompts[i] += output
            model_outputs[i] += output

        print(f"execute {len(codes)} code snippets ...", flush=True)
        batch_results = executor.batch_apply(codes)

        for i, (exec_result, metadata) in zip(code_indices, batch_results):
            exec_result = str(exec_result).strip()
            if len(exec_result) > 100:
                exec_result = exec_result[:50] + "..." + exec_result[-50:]
            runtime_msg = str(metadata['concise_exec_info']).strip() if USE_CONCISE_EXEC_INFO else str(metadata['exec_info']).strip()
            if not exec_result:
                runtime_msg = str(runtime_msg).strip()
                if USE_CONCISE_EXEC_INFO:
                    if len(runtime_msg) > 100:
                        runtime_msg = runtime_msg[:50] + "..." + runtime_msg[-50:]
                    exec_result = runtime_msg
                else:
                    if tokenizer is None:
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH or TOKENIZER_NAME_OR_PATH, trust_remote_code=True)
                    tokens = tokenizer.tokenize(runtime_msg)
                    if len(tokens) > 100:
                        runtime_msg = f"{tokenizer.convert_tokens_to_string(tokens[:50]).strip()} ... {tokenizer.convert_tokens_to_string(tokens[-50:]).strip()}"
                    exec_result = f"Runtime errors: {runtime_msg}"

            prompts[i] += f"\n```output\n{exec_result.strip()}\n```\n"
            model_outputs[i] += f"\n```output\n{exec_result.strip()}\n```\n"

        unfinished_ids = [i for i, is_finished in zip(unfinished_ids, finish_completion) if not is_finished]

        n_iters -= 1

    predictions = [eval(ANSWER_EXTRACTION_FN)(item['messages'][-2]['content'], output, task='interleave') for item, output in tqdm(zip(test_data, model_outputs), desc="extract answer", total=len(model_outputs))]
    program_outputs = [extract_program_output(output) for output in tqdm(model_outputs, desc='extract program output', total=len(model_outputs))]
    assert len(model_outputs) > 0, f"{len(model_outputs)}"

    results = []
    for example, output, pred, program_output in zip(test_data, model_outputs, predictions, program_outputs):
        item = deepcopy(example)
        item.update({
            'model_output': output,
            'prediction': pred,
            'program_output': program_output,
        })
        results.append(item)

    return results

def infer_flip(sample, n_repetition=1):
    test_data = [copy.deepcopy(sample) for _ in range(n_repetition)]
    if PROMPT_FORMAT == 'few_shot':
        assert FEW_SHOT_PROMPT is not None
        prompting = eval(FEW_SHOT_PROMPT)()

    prompts = []
    for example in test_data:
        prompt = ""
        if PROMPT_FORMAT == 'few_shot':
            prompt = prompting.format_prompt(example['messages'][-2]['content'], example['messages'][-1]['content'])
        else:
            for mess in example['messages']:
                if PROMPT_FORMAT == 'sft':
                    if mess['role'] == 'user':
                        prompt += f"User: {mess['content'].strip()}\n\nAssistant:"
                    elif mess['role'] == 'assistant':
                        prompt += mess['content'].strip()
                else:
                    raise NotImplementedError()
            prompt = prompt.lstrip()
        example['prompt'] = prompt
        prompts.append(prompt.lstrip())
    model_outputs = [item['messages'][-1]['content'].strip() for item in test_data]
    unfinished_ids = list(range(len(prompts)))

    executor = PythonExecutor(get_answer_from_stdout=True)

    
    global model, tokenizer
    pbar = tqdm()
    for index in range(n_repetition):
        print(index)
        pbar.update(1)
        unfinished = False
        n_iters = 5
        while n_iters and not unfinished:
            model_inputs = prompts[index]
            finish_completion = None
            print("Loading model and tokenizer...")
            if USE_VLLM:
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME_OR_PATH, trust_remote_code=True)
                    print(f"{'-' * 20} prompt_to_ids {'-' * 20}\n{tokenizer.encode(model_inputs[0])}\n{'-' * 50}", flush=True)
                    print(f"eos_token: {tokenizer.eos_token}", flush=True)
                if model is None:
                    model = LLM(model=MODEL_NAME_OR_PATH, dtype='half', kv_cache_dtype="auto", max_model_len=2048, swap_space=4, gpu_memory_utilization=0.99, enforce_eager=True, tokenizer=TOKENIZER_NAME_OR_PATH, trust_remote_code=True, tensor_parallel_size=1)
                    # model = LLM(model=args.model_name_or_path, dtype='half', kv_cache_dtype="fp8_e4m3", max_model_len=2048, swap_space=4, gpu_memory_utilization=0.4, enforce_eager=True, tokenizer=args.tokenizer_name_or_path, trust_remote_code=True, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
                    # model = LLM(model=args.model_name_or_path, dtype='half', max_model_len=2048, swap_space=4, gpu_memory_utilization=0.4, enforce_eager=True, tokenizer=args.tokenizer_name_or_path, trust_remote_code=True, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
                stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
                if not NO_EXECUTION:
                    stop_words.append("```output")
                if PROMPT_FORMAT == 'few_shot':
                    stop_words.extend(prompting.stop_words())
                outputs = model.generate(model_inputs, SamplingParams(temperature=TEMPERATURE, top_p=1.0, max_tokens=1024, n=1, stop=stop_words))
                outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
                # finish_completion = [output.outputs[0].token_ids[-1] == tokenizer.eos_token_id for output in outputs]
                finish_completion = [output.outputs[0].token_ids[-1] == tokenizer.eos_token_id if (len(output.outputs) > 0 and len(output.outputs[0].token_ids) > 0) else False for output in outputs]
                outputs = [output.outputs[0].text for output in outputs]
            
            # if len(unfinished_ids) != len(outputs):
            #     print(f"input-output mismatch >>> {len(unfinished_ids)} != {len(outputs)}", flush=True)
            #     print(f"----- DEBUG -----\ninputs:\n{model_inputs[:10]}\noutputs:\n{str(outputs[:10])}\n----- DEBUG -----\n", flush=True)
            #     raise RuntimeError()

            if finish_completion is None:
                finish_completion = [finish_answer_prediction(output) for output in outputs]

            print("extract code ...", flush=True)
            codes = []
            code_indices = [0]
            for output, is_finished in zip(outputs, finish_completion):
                output = output.rstrip()
                if not NO_EXECUTION and not is_finished:
                    code = extract_code(model_outputs[index] + output)
                    if code:
                        codes.append(code)
                prompts[index] += output
                model_outputs[index] += output

            print(f"execute {len(codes)} code snippets ...", flush=True)
            batch_results = executor.batch_apply(codes)

            for i, (exec_result, metadata) in zip(code_indices, batch_results):
                exec_result = str(exec_result).strip()
                if len(exec_result) > 100:
                    exec_result = exec_result[:50] + "..." + exec_result[-50:]
                runtime_msg = str(metadata['concise_exec_info']).strip() if USE_CONCISE_EXEC_INFO else str(metadata['exec_info']).strip()
                if not exec_result:
                    runtime_msg = str(runtime_msg).strip()
                    if USE_CONCISE_EXEC_INFO:
                        if len(runtime_msg) > 100:
                            runtime_msg = runtime_msg[:50] + "..." + runtime_msg[-50:]
                        exec_result = runtime_msg
                    else:
                        if tokenizer is None:
                            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH or TOKENIZER_NAME_OR_PATH, trust_remote_code=True)
                        tokens = tokenizer.tokenize(runtime_msg)
                        if len(tokens) > 100:
                            runtime_msg = f"{tokenizer.convert_tokens_to_string(tokens[:50]).strip()} ... {tokenizer.convert_tokens_to_string(tokens[-50:]).strip()}"
                        exec_result = f"Runtime errors: {runtime_msg}"

                prompts[index] += f"\n```output\n{exec_result.strip()}\n```\n"
                model_outputs[index] += f"\n```output\n{exec_result.strip()}\n```\n"

            # unfinished_ids = [i for i, is_finished in zip(unfinished_ids, finish_completion) if not is_finished]

            n_iters -= 1

    predictions = [eval(ANSWER_EXTRACTION_FN)(item['messages'][-2]['content'], output, task='interleave') for item, output in tqdm(zip(test_data, model_outputs), desc="extract answer", total=len(model_outputs))]
    program_outputs = [extract_program_output(output) for output in tqdm(model_outputs, desc='extract program output', total=len(model_outputs))]
    assert len(model_outputs) > 0, f"{len(model_outputs)}"

    results = []
    for example, output, pred, program_output in zip(test_data, model_outputs, predictions, program_outputs):
        item = deepcopy(example)
        item.update({
            'model_output': output,
            'prediction': pred,
            'program_output': program_output,
        })
        results.append(item)

    return results





from config import *
# if args.gpus is not None:
# os.environ['CUDA_VISIBLE_DEVICES'] = GPUS

# print(unparsed_args, flush=True)

model = None
tokenizer = None
pool = None


random.seed(42)


df = pd.read_csv('lv5_sub1_remove_SINGLE.csv')
# midpoint = 1
# df = dfr.iloc[:midpoint]
# df = dfr
print(df.head())
print("#samples in csv:", len(df))
NOTEBOOK_START_TIME = time.time()

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

predictions = []
program_outputs = []
df['model_answer'] = [-1] * len(df)
for i_problem in tqdm(range(len(df))):
    TIME_SPENT = time.time() - NOTEBOOK_START_TIME # accumulated time
    if TIME_SPENT > TIME_LIMIT:
        break
    print(f"\n\n\n########## QUESTION {i_problem} - TIME_SPENT : {TIME_SPENT:.0f} secs")
    problem = df['problem'].loc[i_problem]
    fn = eval(PROCESS_FN)
    item = {}
    item["id"] = str(i_problem)
    # item["level"] = '5',
    # item["category"] = df['id'].loc[i_problem].split('-')[0],
    item["messages"] = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": ""}
    ]
    item = markup_question(None, item, "en", None, TASK)

    now = time.time()
    # results = infer(item, N_REPETITION)
    results = infer_flip(item, N_REPETITION)
    if N_REPETITION > 1:
        _program_outputs = [result['program_output'] for result in results]
        _predictions = [result['prediction'][0] if len(result['prediction']) > 0 else -1 for result in results]
        print(_program_outputs)
        print(_predictions)
        _program_output = Counter(_program_outputs).most_common()[0][0]
        _prediction = Counter(_predictions).most_common()[0][0]
    else:
        _program_output = results[0]['program_output']
        _prediction = results[0]['prediction']
    predictions.append(_prediction)
    program_outputs.append(_program_output)
    # print("problem")
    # print(item['prediction'])
    # print(item['program_output'])
    # item['prediction'] = [Counter(item['prediction']).most_common()[0][0]]
    # item['program_output'] = Counter(item['program_output']).most_common()[0][0]
    # print(item['prediction'])
    # print(item['program_output'])
    try:
        df['model_answer'].iloc[i_problem] = round(float(eval(_prediction))) % 1000
    except:
        df['model_answer'].iloc[i_problem] = -1
    df['match'] = df.answer == df.model_answer
    print(f'{df.match.sum()} matches in {i_problem+1} examples')

df['model_answer'] = predictions
df['match'] = df.answer == df.model_answer
df.to_csv("submission.csv", header=True, index=False)
print(f'{df.match.sum()} matches in {len(df)} examples')

print(df)
