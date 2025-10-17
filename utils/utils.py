
import re
import random
import torch
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA if using GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

def topk_indices(arr, k):
    """
    Returns the top-K values and their (x, y) indices from a 2D NumPy array.

    Args:
        arr (np.ndarray): 2D input array.
        k (int): Number of top elements to retrieve.

    Returns:
        list: List of (value, x, y) tuples sorted in descending order.
    """
    flat_indices = np.argpartition(arr.ravel(), -k)[-k:]  # Get indices of top-K elements (unordered)
    sorted_indices = flat_indices[np.argsort(arr.ravel()[flat_indices])][::-1]  # Sort them in descending order
    
    topk_coords = [(arr.flat[i], i // arr.shape[1], i % arr.shape[1]) for i in sorted_indices]  # Convert to (value, x, y)
    return topk_coords

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

def calculate_token_cost(results, tokenizer):
    token_budget = []
    for output in results:
        token_cost = tokenizer.encode(output[0])
        token_budget.append(len(token_cost))
    print(f"Samples: {len(token_budget)} Total token cost: {np.mean(token_budget)}")

