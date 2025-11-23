import json
import os
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

def compute_pass_at_k(base_dir, k_values=[1, 4, 8, 16], model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", mode="seed"):
    """
    Compute Pass@K metrics by merging results from K different configurations.
    If any configuration passes a sample, that sample is considered correct.
    
    Args:
        base_dir: Base directory containing configuration folders
        k_values: List of K values to compute Pass@K for
        model_name: Model name for tokenizer initialization
        mode: "seed" for different seeds, "coef" for different numb+coef combinations
    """
    
    # Initialize tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find all configuration directories based on mode
    if mode == "seed":
        config_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith('seed')])
        config_type = "seed"
    elif mode == "coef":
        config_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith('Steering')])
        config_type = "numb+coef combination"
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'seed' or 'coef'")
    
    if not config_dirs:
        print(f"No {config_type} directories found in {base_dir}")
        return
    
    print(f"Found {len(config_dirs)} {config_type} directories")
    
    # Load all evaluation results
    all_results = {}
    for config_dir in config_dirs:
        if mode == "seed":
            # For seed mode: base_dir/seed{N}/math_eval.jsonl
            eval_file = config_dir / "math_eval.jsonl"
        elif mode == "coef":
            # For coef mode: base_dir/{config}/mix_others_low_rank_1000/.../MATH500/seed{N}/math_eval.jsonl
            # Find the seed directory inside the config
            math500_dir = config_dir / "mix_others_low_rank_1000" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-1.5B" / "MATH500"
            if not math500_dir.exists():
                print(f"Warning: {math500_dir} not found")
                continue
            # Find the first seed directory
            seed_dirs_in_config = [d for d in math500_dir.iterdir() if d.is_dir() and d.name.startswith('seed')]
            if not seed_dirs_in_config:
                print(f"Warning: No seed directory found in {math500_dir}")
                continue
            eval_file = seed_dirs_in_config[0] / "math_eval.jsonl"
        
        if not eval_file.exists():
            print(f"Warning: {eval_file} not found")
            continue
            
        config_name = config_dir.name
        all_results[config_name] = []
        
        with open(eval_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_results[config_name].append(data)
    
    if not all_results:
        print("No evaluation results found")
        return
    
    # Get total number of samples (assuming all seeds have same samples)
    num_samples = len(next(iter(all_results.values())))
    print(f"Total samples: {num_samples}")
    
    # Compute Pass@K for each K value
    results_summary = {}
    
    for k in k_values:
        if k > len(config_dirs):
            print(f"Warning: K={k} exceeds number of available {config_type}s ({len(config_dirs)})")
            continue
        
        # For each sample, check if any of the K configurations pass it
        passed_samples = 0
        token_budget = []
        
        for sample_idx in range(num_samples):
            # Check first K configurations for this sample
            sample_passed = False
            sample_tokens = []
            
            for config_idx, config_name in enumerate(sorted(all_results.keys())[:k]):
                sample_data = all_results[config_name][sample_idx]
                
                # Calculate token length for this generation
                if 'model_generation' in sample_data:
                    if isinstance(sample_data['model_generation'], list):
                        text = sample_data['model_generation'][0]
                    else:
                        text = sample_data['model_generation']
                    
                    token_cost = tokenizer.encode(text)
                    sample_tokens.append(len(token_cost))
                
                # Check if any evaluation result is True
                if 'all_eval' in sample_data and any(sample_data['all_eval']):
                    sample_passed = True
                    break
            
            if sample_passed:
                passed_samples += 1
            
            # Add average token length across K seeds for this sample
            if sample_tokens:
                token_budget.append(sum(sample_tokens) / len(sample_tokens))
        
        pass_rate = (passed_samples / num_samples) * 100
        avg_tokens = sum(token_budget) / len(token_budget) if token_budget else 0
        
        results_summary[f"Pass@{k}"] = {
            "passed": passed_samples,
            "total": num_samples,
            "rate": pass_rate,
            "avg_token_length": avg_tokens
        }
        
        print(f"Pass@{k}: {passed_samples}/{num_samples} = {pass_rate:.2f}% | Avg Tokens: {avg_tokens:.2f}")
    
    # Save summary
    output_filename = f"pass_at_k_summary_{mode}.json"
    output_file = Path(base_dir) / output_filename
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nSummary saved to: {output_file}")
    
    return results_summary

if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Mode 1: seedPassK - different seeds with same configuration
    # print("\n" + "=" * 80)
    # print("Mode 1: Seed-based Pass@K (different seeds, same numb+coef)")
    # print("=" * 80)
    # seed_base_dir = "/home/charlie/CREST/rebuttal/seedPassK/SteeringTrue_numb64_coef-1_modeafter_o_proj_correct_0_100/mix_others_low_rank_1000/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/MATH500"
    # compute_pass_at_k(seed_base_dir, k_values=[1, 4, 8, 16], model_name=model_name, mode="seed")
    
    # Mode 2: coefPassK - different configurations with same seed
    print("\n" + "=" * 80)
    print("Mode 2: Coef-based Pass@K (same seed, different numb+coef combinations)")
    print("=" * 80)
    coef_base_dir = "/home/charlie/CREST/rebuttal/coefPassK"
    compute_pass_at_k(coef_base_dir, k_values=[1, 4, 8, 16], model_name=model_name, mode="coef")