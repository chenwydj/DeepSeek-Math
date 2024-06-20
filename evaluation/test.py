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
from concurrent.futures import TimeoutError
from eval.utils import generate_completions, load_hf_lm_and_tokenizer
from eval.python_executor import PythonExecutor
from transformers import AutoTokenizer

from data_processing.answer_extraction import *
from data_processing.process_utils import *
from eval.eval_script import *
from few_shot_prompts import *

from infer.config import *
# if args.gpus is not None:
# os.environ['CUDA_VISIBLE_DEVICES'] = GPUS

# print(unparsed_args, flush=True)


model = LLM(model=MODEL_NAME_OR_PATH, dtype='half', kv_cache_dtype="auto", max_model_len=2048, swap_space=4, gpu_memory_utilization=0.99, enforce_eager=True, tokenizer=TOKENIZER_NAME_OR_PATH, trust_remote_code=True, tensor_parallel_size=1)
# model = LLM(model=MODEL_NAME_OR_PATH, dtype='half', kv_cache_dtype="fp8_e5m2", max_model_len=2048, swap_space=4, gpu_memory_utilization=0.99, enforce_eager=True, tensor_parallel_size=1)