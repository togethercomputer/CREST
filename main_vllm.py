
import os
from probing.modeling_utils.vllm.qwen2.monkey_patch import monkey_patch_qwen2_vllm
from probing.modeling_utils.vllm.qwen3.monkey_patch import monkey_patch_qwen3_vllm
from probing.modeling_utils.vllm.qwen3_moe.monkey_patch import monkey_patch_qwen3_moe_vllm
from probing.modeling_utils.vllm.gpt_oss.monkey_patch import monkey_patch_gpt_oss_vllm
# due to TP, we have to patch here

if os.environ.get("STEERING", "False") == "True":
    # get model name
    model_name = os.environ.get("MODEL_NAME_OR_PATH")
    if "DeepSeek-R1-Distill-Qwen" in model_name:
        monkey_patch_qwen2_vllm(os.environ.get("STEERING_VECTOR_PATH"), os.environ.get("STEERING_NUMBER"), os.environ.get("STEERING_COEF"), os.environ.get("STEERING_MODE"), os.environ.get("MODEL_NAME_OR_PATH"))
    elif "Qwen3-4B" in model_name:
        monkey_patch_qwen3_vllm(os.environ.get("STEERING_VECTOR_PATH"), os.environ.get("STEERING_NUMBER"), os.environ.get("STEERING_COEF"), os.environ.get("STEERING_MODE"), os.environ.get("MODEL_NAME_OR_PATH"))
    elif "Qwen3-30B" in model_name:
        monkey_patch_qwen3_moe_vllm(os.environ.get("STEERING_VECTOR_PATH"), os.environ.get("STEERING_NUMBER"), os.environ.get("STEERING_COEF"), os.environ.get("STEERING_MODE"), os.environ.get("MODEL_NAME_OR_PATH"))
    elif "gpt-oss" in model_name:
        monkey_patch_gpt_oss_vllm(os.environ.get("STEERING_VECTOR_PATH"), os.environ.get("STEERING_NUMBER"), os.environ.get("STEERING_COEF"), os.environ.get("STEERING_MODE"), os.environ.get("MODEL_NAME_OR_PATH"))
    else:
        raise ValueError(f"Model name not supported: {model_name}")
    
import re
import json
import random
import torch
import evaluate
import argparse

from collections import Counter
from datasets import load_dataset, Dataset
from functools import partial


from transformers import AutoTokenizer
from datasets import load_dataset

import sys
import os
import gc
from typing import Any
from lighteval.tasks.extended.lcb.codegen_metrics import translate_private_test_cases, codegen_metrics, extract_code

from probing.get_omni_results import main as eval_main
from vllm import LLM, SamplingParams
    
os.environ["TOKENIZERS_PARALLELISM"] = "false"
exact_match = evaluate.load("exact_match")

import numpy as np
def hour_to_num(hr_str):
    return float(hr_str.split(':')[0]) + (
        0.5 if hr_str.split(':')[1] == '30' else 0.0
    )

def _parse_response(response: str):
    """Parse the response.

    Returns a parsed suggested meeting time in (day, start_hour, end_hour).

    Args:
        response: Raw response from the model.

    Returns:
        A tuple of (day, start_hour, end_hour).
    """
    time_strs = re.findall(r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+', response)
    if not time_strs:
        return '', -1, -1
    # If multiple matches are found, return the first one.
    time_str = time_strs[0]
    day, hour_str = (
        time_str.split(',')[0].strip(),
        time_str.split(',')[1].strip(),
    )
    start_hour, end_hour = (
        hour_str.split('-')[0].strip(),
        hour_str.split('-')[1].strip(),
    )
    return day, hour_to_num(start_hour), hour_to_num(end_hour)

def compute_solve_rate(responses: list[str], solutions: list[str]):
    """Computes solve rate by comparing model responses to golden solutions.

    Args:
        responses: A list of model responses.
        solutions: The corresponding list of golden solutions for the same tasks.

    Returns:
        A scalr solve rate.
    """
    solved_count = 0

    for r, s in zip(responses, solutions):
        r_day, r_start_hour, r_end_hour = _parse_response(r)
        s_day, s_start_hour, s_end_hour = _parse_response(s)
        if (
            r_day == s_day
            and r_start_hour == s_start_hour
            and r_end_hour == s_end_hour
        ):
            solved_count += 1
    return float(solved_count) / len(responses)

def metrics_plan(output):
    """Compute metrics for the calendar scheduling task.

    Args:
        output: A list of dictionaries containing model responses and golden solutions.

    Returns:
        A dictionary containing the solve rate.
    """
    responses = [item['output_text'] for item in output]
    solutions = [item['golden_plan'] for item in output]

    overall_solved_rate = compute_solve_rate(responses, solutions)
    print(f'Overall solve rate: {overall_solved_rate}')

    return {'solve_rate': overall_solved_rate}

def extract_choice(response: str) -> str:
    """
    Extracts 'Answer: X' (where X in [A-D]) from the last line of the model's response.
    Returns the extracted letter or 'INVALID' if not found.
    """
    match = re.search(r"(?i)^answer:\s*([A-D])\s*$", response.strip().splitlines()[-1])
    if match:
        return match.group(1).upper()
    return "INVALID"

def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits

def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output

def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a

def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA if using GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

def main(args):
    set_seed(args.seed)
    print("Loading data...")
    test_data = []
    if args.dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            gt = extract_box(example["solution"])
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt":gt,
            })
    elif args.dataset in ["MATH", "MATH_train", "MATH_subset"]:
        if args.dataset == "MATH_train":
            data_path = "data/MATH/train.jsonl"
        elif args.dataset == "MATH_subset":
            data_path = "data/MATH/test_subset.jsonl"
        else:
            data_path = "data/MATH/test.jsonl"
        
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                # pattern = r"\\boxed\{([^\}]+)\}"
                # gt = re.findall(pattern, example["solution"])[-1].strip()
                gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":gt,
                })
    elif args.dataset == "AIME":
        data = load_dataset("AI-MO/aimo-validation-aime", split="train")
        for example in data:
            gt = example["answer"]
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt":gt,
            })
    elif args.dataset == "AIME2024":
        data = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        for example in data:
            gt = str(example["Answer"])
            test_data.append({
                "question": example["Problem"],
                "answer": example["Solution"],
                "gt":gt,
            })
    elif args.dataset in ["GSM", "GSM_train"]:
        if args.dataset == "GSM_train":
            data_path = "probing/data/gsm/train.jsonl"
        else:
            data_path = "probing/data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer": answer
                })
    elif args.dataset == "amc23":
        data = load_dataset("math-ai/amc23", split="test")
        for example in data:
            gt = str(example["answer"])
            test_data.append({
                "question": example["question"],
                "answer": example["answer"],
                "gt":gt,
            })
    elif args.dataset == "aime25":
        data = load_dataset("math-ai/aime25", split="test")
        for example in data:
            gt = str(example["answer"])
            test_data.append({
                "question": example["problem"],
                "answer": example["answer"],
                "gt":gt,
            })
    elif args.dataset == "lcb":
        data = "livecodebench/code_generation_lite"
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag='v5', split='test', trust_remote_code=True)
        
        test_data = []
        def prepare_prompt(line: dict[str, Any]) -> str:
            query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
            query += f"Question: {line['question_content']}\n\n"
            
            if starter_code := line.get("starter_code", None):
                query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
                query += f"```python\n{starter_code}\n```\n\n"
            else:
                query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
                query += "```python\n# YOUR CODE HERE\n```\n\n"
            return query
        
        for data in dataset:
            query = prepare_prompt(data)
            public_test_cases = json.loads(data["public_test_cases"])
            private_test_cases = translate_private_test_cases(data["private_test_cases"])
            conversations = [
                {"role": "user", "content": query},
            ]
            data["conversations"] = conversations
            data["query"] = query
            data["inputs"] = [test["input"] for test in public_test_cases + private_test_cases]
            data["outputs"] = [test["output"] for test in public_test_cases + private_test_cases]
            data["fn_name"] = json.loads(data["metadata"]).get("func_name", None)

            test_data.append(data)

    elif args.dataset == "gpqa":
        dataset = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train')

        INDEX_MAP = {
            0: "A",
            1: "B",
            2: "C",
            3: "D"
        }

        GPQA_QUERY_TEMPLATE = """
        Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

        {Question}

        A) {A}
        B) {B}
        C) {C}
        D) {D}
        """.strip()

        for data in dataset:
            gold_index = random.randint(0, 3)
            choices = [data["Incorrect Answer 1"], data["Incorrect Answer 2"], data["Incorrect Answer 3"]]
            choices.insert(gold_index, data["Correct Answer"])
            data['gold_index'] = INDEX_MAP[gold_index]
            data['choices'] = choices

            query = GPQA_QUERY_TEMPLATE.format(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=data["Question"]
            )

            conversations = [
                {"role": "user", "content": query}
            ]
            data["conversations"] = conversations
            test_data.append(data)
    
    elif args.dataset == "cp":
        data_path = "probing/data/PLAN/calendar_scheduling.json"
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        records = []
        for sample_id, sample_data in dataset.items():
            sample_data["id"] = sample_id
            records.append(sample_data)
        dataset = Dataset.from_list(records)

        for data in dataset:
            conversations = [
                {"role": "user", "content": data['prompt_5shot']}
            ]
            data["conversations"] = conversations
            test_data.append(data)

    else:
        raise ValueError("Dataset not supported")

    if args.max_examples and len(test_data) > args.max_examples:
        if args.max_examples == -1:
            args.max_examples = len(test_data)
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

    # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    
    prompts = []
    for i, example in enumerate(test_data):
        if args.use_chat_format:
            if args.dataset == "lcb" or args.dataset == "gpqa" or args.dataset == "cp":
                messages = example["conversations"]
            elif "gemma" in args.model_name_or_path or "deepseek" in args.model_name_or_path:
                messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
            else:
                messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
            if "gpt-oss" in args.model_name_or_path:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort="high", skip_special_tokens = False)
            else:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        else:
            prompt =  prefix + "Question: " + example["question"].strip() + "\nAnswer: "
        prompts.append(prompt)
    
    print("enforce_eager", args.enforce_eager)
    model = LLM(model=args.model_name_or_path, download_dir=os.environ.get("HF_HOME", ''),
        tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, 
        swap_space=16, 
        gpu_memory_utilization=0.90, 
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_tokens,
        enforce_eager=args.enforce_eager)
    
    if not args.logit_adjustment:
        sampling_params = SamplingParams(n=args.n_samples,
                                        temperature=args.temperature,
                                        top_p=0.95,
                                        max_tokens=args.max_tokens)
    else:
        vocab = tokenizer.get_vocab()
        logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]).to("cuda")
        logit_adjustment_process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=args.n_samples,
                                        temperature=args.temperature,
                                        max_tokens=args.max_tokens,
                                        top_p=0.95,
                                        logits_processors=[logit_adjustment_process]
                                        )
    
    if not args.steering:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
        result = []
        for output in outputs:
            attempts = []
            for ith_output in output.outputs:
                attempts.append(ith_output.text)
            result.append(attempts)
    else:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
        result = []
        for output in outputs:
            attempts = []
            for ith_output in output.outputs:
                attempts.append(ith_output.text)
            result.append(attempts)

        # force maintain batch size otherwise we will have continuous batching
        # batch_size = args.batch_size
        # outputs = []
        # for i in range(0, len(prompts), batch_size):
        #     batch_prompts = prompts[i:i+batch_size]
        #     print("process {} / {}".format(i, len(prompts)))
        #     output = model.generate(prompts=batch_prompts, sampling_params=sampling_params)                
        #     outputs.append(output)

        # result = []
        # for single_output in outputs:
        #     for output in single_output:
        #         attempts = []
        #         for ith_output in output.outputs:
        #             attempts.append(ith_output.text)
        #         result.append(attempts)

    outputs = [[trim_output(o) for o in output] for output in result]
    if args.dataset in ["GSM", "GSM_train"]:
        predictions = []
        for output in outputs:
            answer = []
            # replace numbers like `x,xxx` with `xxxx`
            for o in output:
                if "boxed" in o:
                    ans = extract_box(o)
                else:
                    ans = extract_last_number(o)
                if ans is not None:
                    answer.append(ans)
            if len(answer) == 0:
                answer = o 
            else:
                counter = Counter(answer)
                answer = counter.most_common(1)[0][0]
            predictions.append(answer)
            

        print("Calculating accuracy...")
        targets = [example["answer"] for example in test_data]

        em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
        print(f"Exact match : {em_score}")

        predictions = [{
            "question": example["question"],
            "answer": example["answer"],
            "prompt": prompt,
            "model_generation": output,
            "prediction": pred
        } for example, output, pred, prompt in zip(test_data, outputs, predictions, prompts)]

        with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
            for prediction in predictions:
                fout.write(json.dumps(prediction) + "\n")

        with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
            json.dump({
                "exact_match": em_score
            }, fout, indent=4)
    elif args.dataset == "gpqa":
        results = []
        predictions = []
        for output in outputs:
            for o in output:
                predictions.append(o)
        for data, output in zip(test_data, predictions):
            golds = data['gold_index']
            predictions = output
            predictions_answer = extract_choice(output)
            if predictions_answer == 'INVALID':
                result = {"extractive_match": 0.0}
            else:
                result = {"extractive_match": 1.0 if predictions_answer == golds else 0.0}
            results.append(
                {   
                    "score": result["extractive_match"] * 100,
                    "model_generation": [output],
                    "choices": golds,
                    "query": data['Question']
                }
            )
        total_accuracy = 0.0
        count = 0
        unique_result = {}

        for item in results:
            total_accuracy += item['score']
            count += 1
            
            if item['query'] not in unique_result:
                unique_result[item['query']] = item["score"]
            else:
                unique_result[item['query']] = max(item["score"], unique_result[item['query']])

        with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
            for prediction in results:
                fout.write(json.dumps(prediction) + "\n")

        with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
            json.dump({
                "pass@1": total_accuracy / count if count > 0 else 0
            }, fout, indent=4)
    elif args.dataset == "cp":
        results = []
        predictions = []
        for data, output in zip(test_data, outputs):
            for o in output:
                results.append(
                    {
                        "golden_plan": data['golden_plan'],
                        "output_text": o
                    }
                )
                predictions.append(
                    {
                        "model_generation": [o],
                        "golden_plan": data['golden_plan'],
                    }
                )
        rate = metrics_plan(results)
        
        with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
            for prediction in predictions:
                fout.write(json.dumps(prediction) + "\n")

        with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
            json.dump({
                "solve_rate": rate
            }, fout, indent=4)

    elif args.dataset == "lcb":
        all_code_snippets = []
        all_evaluation_sample = []
        predictions = []
        for output, data in zip(outputs, test_data):
            for o in output:
                extracted_code = [[extract_code(o)]]
                eval_sample = {
                    "inputs": data["inputs"],
                    "outputs": data["outputs"],
                    "fn_name": data["fn_name"],
                }
                evaluation_sample = [{"input_output": json.dumps(eval_sample)}]
                all_code_snippets.extend(extracted_code)
                all_evaluation_sample.extend(evaluation_sample)
                predictions.append(
                    {
                        "model_generation": [o],
                        "inputs": data["inputs"],
                        "outputs": data["outputs"],
                        "fn_name": data["fn_name"],
                    }
                )
        metrics, _ = codegen_metrics(
                        all_evaluation_sample,
                        all_code_snippets,
                        k_list=[1],
                        num_process_evaluate=64,
                        timeout=20
                )
                
        print("Metrics:", metrics)
        with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
            for prediction in predictions:
                fout.write(json.dumps(prediction) + "\n")

        with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
            json.dump({
                "lighteval": metrics
            }, fout, indent=4)
    else:
        predictions = [{
            "prompt": prompt,
            "problem": example["question"],
            "answer": example["gt"],
            "solution":  example["answer"],
            "model_generation": output,
        } for example, output, prompt in zip(test_data, outputs, prompts)]

        with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
            for prediction in predictions:
                fout.write(json.dumps(prediction) + "\n")
    
        eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="If given, we will use the test set."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=['Wait', 'Alternatively']
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--avoid_overthink",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--steering",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--steering_vector_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--steering_number",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--steering_mode",
        type=str,
        default=None
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        default=False
    )
    args = parser.parse_args()
    if args.temperature==0.0:
        args.n_samples = 1

    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)

    main(args)
    
