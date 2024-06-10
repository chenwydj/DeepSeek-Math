# python infer/run_tool_integrated_eval.py --data_dir outputs/DeepSeekMath-RL-debug/math-test/infer_logs/tool/test_data --max_num_examples 100000000000000 --save_dir outputs/DeepSeekMath-RL-debug/math-test/infer_logs/tool/samples --model deepseek-ai/deepseek-math-7b-rl --tokenizer deepseek-ai/deepseek-math-7b-rl --eval_batch_size 1 --temperature 0.9 --repeat_id_start 0 --n_repeat_sampling 1 --n_repetition 15 --n_subsets 1 --prompt_format sft --few_shot_prompt None --answer_extraction_fn extract_math_answer --eval_fn eval_math --subset_id 0 --gpus 0  --use_vllm

DATA_DIR = "outputs/DeepSeekMath-RL-debug/math-test/infer_logs/tool/test_data"
MAX_NUM_EXAMPLES = 100000000000000 # "maximum number of examples to evaluate.")
SAVE_DIR = "outputs/DeepSeekMath-RL-debug/math-test/infer_logs/tool/samples"
MODEL_NAME_OR_PATH = "deepseek-ai/deepseek-math-7b-rl" # "if specified, we will load the model to generate the predictions.")
TOKENIZER_NAME_OR_PATH = "deepseek-ai/deepseek-math-7b-rl" # "if specified, we will load the tokenizer from here.")
EVAL_BATCH_SIZE = 1 # "batch size for evaluation.")
LOAD_IN_8BIT = False # "load model in 8bit mode, which will reduce memory and speed up inference.")
GPTQ = False # If given, we're evaluating a 4-bit quantized GPTQ model.")
USE_VLLM = True
LOAD_IN_HALF = False
INFER_TRAIN_SET = False
N_SUBSETS = 1
SUBSET_ID = 0
TEMPERATURE = 0.9
REPEAT_ID_START = 0
N_REPEAT_SAMPLING = 1
N_REPETITION = 15
COMPLETE_PARTIAL_OUTPUT = False
USE_CONCISE_EXEC_INFO = False
PROMPT_FORMAT = 'sft' # , choices=['sft', 'few_shot'], default='sft')
FEW_SHOT_PROMPT = None
ANSWER_EXTRACTION_FN = "extract_math_answer"
NO_EXECUTION = False
EVAL_FN = "eval_math"
GPUS = 0

TIME_LIMIT = 9*60*60 - 20*60 # 9h is 32400s

TEST_PATH = "./lv5_remove.csv"
LANGUAGE = "en"
TASKS = "tool"
PROCESS_FN = "process_math_test_csv"
ANSWER_EXTRACTION_FN = "extract_math_answer"
EVAL_FN = "eval_math"