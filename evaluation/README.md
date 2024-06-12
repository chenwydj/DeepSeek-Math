```bash
python run_subset_parallel.py --output-dir outputs/DeepSeekMath-RL-debug --model-path deepseek-ai/deepseek-math-7b-rl --tokenizer-path deepseek-ai/deepseek-math-7b-rl --model-size 7b --use-vllm --test-conf configs/zero_shot_test_configs.json --test-conf configs/zero_shot_test_configs.json --n-repeats 1 --temperature 0.9 --ngpus 1 --rank 0 --n_repetitions 3
Evaluating deepseek-ai/deepseek-math-7b-rl

python infer/run_tool_integrated_eval.py --data_dir outputs/DeepSeekMath-RL-debug/math-test/infer_logs/tool/test_data --max_num_examples 100000000000000 --save_dir outputs/DeepSeekMath-RL-debug/math-test/infer_logs/tool/samples --model deepseek-ai/deepseek-math-7b-rl --tokenizer deepseek-ai/deepseek-math-7b-rl --eval_batch_size 1 --temperature 0.9 --repeat_id_start 0 --n_repeat_sampling 1 --n_subsets 1 --prompt_format sft --few_shot_prompt None --answer_extraction_fn extract_math_answer --eval_fn eval_math --subset_id 0 --gpus 0  --use_vllm

TOKENIZERS_PARALLELISM=true python infer/run_tool_integrated_eval.py

n_repetition = 18
output acc = 42.00000; program acc = 39.33333
TIME SPENT >>> 3996.6007463932037 sec.

n_repetition = 20, timeout = 20, n_iter = 3
150 problems 20011 sec.
```



## 1. Introduction

We provide a test script for both zero-shot and few-shot evaluation on mathematical reasoning benchmarks used in our paper.

## 2. Setup

First configure the `prefix` in `environment.yml` and then run the following command
```
conda env create -f environment.yml
```

## 3. Evaluation

For chain-of-thought evaluation of DeepSeekMath-Instruct and DeepSeekMath-RL, our script (see `def markup_question()` in `run_subset_parallel.py`) processes each question as follows:
* English questions: `{question}\nPlease reason step by step, and put your final answer within \\boxed{}.`
* Chinese questions: `{question}\n请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。`

For tool-integrated reasoning, we process each question as follows:
* English questions: `{question}\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.`
* Chinese questions: `{question}\n请结合自然语言和Python程序语言来解答问题，并把最终答案放置于\\boxed{}中。`

We provide an example of testing the DeepSeekMath-Base 7B using 8 GPUs.

If you wish to use a different model or dataset, you can modify the configs in `submit_eval_jobs.py` and `configs/*test_configs.json`

```
python submit_eval_jobs.py --n-gpus 8
```

Wait for all processes to finish, and then run the following command to aggregate results from all processes

```
python summarize_results.py [--eval-atp]
```
where the option `--eval-atp` will invoke `unsafe_score_minif2f_isabelle.py` to evaluate the informal-to-formal proving results. Please make sure you have set up the [PISA](https://github.com/wellecks/lm-evaluation-harness/blob/minif2f-isabelle/docs/isabelle_setup.md) server before using this option.

A summary of all evaluation results will be saved as `evaluation_results.json`

## 4. Model Outputs

We provide all model outputs in `outputs.zip`.
