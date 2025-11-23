import json
import os
from pathlib import Path
from collections import defaultdict
import re

def analyze_predictions(jsonl_file):
    """
    Analyze predictions.jsonl to find longest responses, failures, and strange cases.
    
    Args:
        jsonl_file: Path to predictions.jsonl file
    """
    
    predictions = []
    
    # Load all predictions
    print(f"Loading predictions from: {jsonl_file}")
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                predictions.append({
                    'line_num': line_num,
                    'score': data.get('score', 0),
                    'model_generation': data.get('model_generation', []),
                    'choices': data.get('choices', ''),
                    'query': data.get('query', '')[:100]  # First 100 chars
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    
    print(f"Loaded {len(predictions)} predictions\n")
    
    # Calculate text lengths
    for pred in predictions:
        if isinstance(pred['model_generation'], list) and pred['model_generation']:
            text = pred['model_generation'][0]
        else:
            text = pred['model_generation']
        
        pred['text_length'] = len(text)
        pred['text'] = text
        pred['word_count'] = len(text.split())
        pred['line_count'] = text.count('\n') + 1
        
        # Check for strange patterns
        pred['has_repetition'] = check_repetition(text)
        pred['has_long_think'] = '<think>' in text.lower() or '</think>' in text.lower()
        pred['think_ratio'] = calculate_think_ratio(text)
    
    # Analysis 1: Top 10 longest responses
    print("=" * 80)
    print("TOP 10 LONGEST RESPONSES (by character count)")
    print("=" * 80)
    sorted_by_length = sorted(predictions, key=lambda x: x['text_length'], reverse=True)
    
    for i, pred in enumerate(sorted_by_length[:10], 1):
        print(f"\n{i}. Line {pred['line_num']}")
        print(f"   Score: {pred['score']}")
        print(f"   Length: {pred['text_length']:,} chars | {pred['word_count']:,} words | {pred['line_count']:,} lines")
        print(f"   Query: {pred['query']}...")
        if pred['has_repetition']:
            print(f"   ⚠️  WARNING: Contains repetition!")
        if pred['has_long_think']:
            print(f"   💭 Contains <think> tags | Think ratio: {pred['think_ratio']:.1%}")
    
    # Analysis 2: Failure cases (score = 0)
    print("\n" + "=" * 80)
    print("FAILURE CASES (score = 0)")
    print("=" * 80)
    failures = [p for p in predictions if p['score'] == 0]
    print(f"Total failures: {len(failures)} / {len(predictions)} ({len(failures)/len(predictions)*100:.1f}%)\n")
    
    # Show top 5 longest failures
    failures_sorted = sorted(failures, key=lambda x: x['text_length'], reverse=True)
    print("Top 5 longest failure cases:")
    for i, pred in enumerate(failures_sorted[:5], 1):
        print(f"\n{i}. Line {pred['line_num']}")
        print(f"   Length: {pred['text_length']:,} chars | {pred['word_count']:,} words")
        print(f"   Query: {pred['query']}...")
        if pred['has_repetition']:
            print(f"   ⚠️  WARNING: Contains repetition!")
    
    # Analysis 3: Strange cases
    print("\n" + "=" * 80)
    print("STRANGE CASES")
    print("=" * 80)
    
    # Case 1: Extremely long responses (>50k chars)
    extremely_long = [p for p in predictions if p['text_length'] > 50000]
    print(f"\n1. Extremely long responses (>50k chars): {len(extremely_long)}")
    for pred in extremely_long[:5]:
        print(f"   Line {pred['line_num']}: {pred['text_length']:,} chars | Score: {pred['score']}")
    
    # Case 2: Responses with repetition
    with_repetition = [p for p in predictions if p['has_repetition']]
    print(f"\n2. Responses with detected repetition: {len(with_repetition)}")
    for pred in with_repetition[:5]:
        print(f"   Line {pred['line_num']}: {pred['text_length']:,} chars | Score: {pred['score']}")
    
    # Analysis 4: Statistics summary
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    avg_length = sum(p['text_length'] for p in predictions) / len(predictions)
    avg_words = sum(p['word_count'] for p in predictions) / len(predictions)
    avg_score = sum(p['score'] for p in predictions) / len(predictions)
    
    print(f"Average response length: {avg_length:,.0f} chars | {avg_words:,.0f} words")
    print(f"Average score: {avg_score:.1f}")
    print(f"Max response length: {max(p['text_length'] for p in predictions):,} chars")
    print(f"Min response length: {min(p['text_length'] for p in predictions):,} chars")
    print(f"Success rate: {sum(1 for p in predictions if p['score'] > 0)/len(predictions)*100:.1f}%")
    
    # Save detailed analysis to CREST directory
    output_file = Path("/home/charlie/CREST/analysis_report.txt")
    save_detailed_report(predictions, sorted_by_length, failures, output_file)
    print(f"\n📄 Detailed report saved to: {output_file}")
    
    # Also save the longest failure case for inspection
    if failures:
        longest_failure = max(failures, key=lambda x: x['text_length'])
        failure_file = Path("/home/charlie/CREST/longest_failure.txt")
        with open(failure_file, 'w', encoding='utf-8') as f:
            f.write(f"LONGEST FAILURE CASE (Line {longest_failure['line_num']})\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Score: {longest_failure['score']}\n")
            f.write(f"Length: {longest_failure['text_length']:,} chars\n")
            f.write(f"Query: {longest_failure['query']}\n\n")
            f.write("Full Response:\n")
            f.write("-" * 80 + "\n")
            f.write(longest_failure['text'])
        print(f"📄 Longest failure saved to: {failure_file}")
    
    return predictions, sorted_by_length, failures

def check_repetition(text, window=100):
    """Check if text contains repetitive patterns"""
    if len(text) < window * 2:
        return False
    
    # Check for exact repetition of chunks
    for i in range(0, len(text) - window, window // 2):
        chunk = text[i:i+window]
        rest = text[i+window:]
        if chunk in rest:
            # Check if it appears multiple times
            count = rest.count(chunk)
            if count >= 3:  # Repeated at least 3 times
                return True
    
    return False

def calculate_think_ratio(text):
    """Calculate ratio of content inside <think> tags"""
    # Simple heuristic: check for <think> tags
    if '<think>' not in text.lower():
        return 0.0
    
    # Try to extract think content
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        return 0.0
    
    think_length = sum(len(m) for m in matches)
    total_length = len(text)
    
    return think_length / total_length if total_length > 0 else 0.0

def save_detailed_report(predictions, sorted_by_length, failures, output_file):
    """Save a detailed analysis report"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PREDICTIONS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Top 20 longest responses
        f.write("TOP 20 LONGEST RESPONSES\n")
        f.write("-" * 80 + "\n")
        for i, pred in enumerate(sorted_by_length[:20], 1):
            f.write(f"\n{i}. Line {pred['line_num']}\n")
            f.write(f"   Score: {pred['score']}\n")
            f.write(f"   Length: {pred['text_length']:,} chars | {pred['word_count']:,} words\n")
            f.write(f"   Query: {pred['query']}...\n")
            f.write(f"   Answer: {pred['choices']}\n")
            if pred['has_repetition']:
                f.write(f"   ⚠️  Contains repetition\n")
            if pred['think_ratio'] > 0:
                f.write(f"   Think ratio: {pred['think_ratio']:.1%}\n")
            
            # Save first 500 chars of the text
            preview = pred['text'][:500]
            f.write(f"\n   Preview:\n")
            for line in preview.split('\n')[:10]:
                f.write(f"   {line}\n")
            f.write("\n")
        
        # All failures
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("ALL FAILURE CASES (Score = 0)\n")
        f.write("-" * 80 + "\n")
        for i, pred in enumerate(failures, 1):
            f.write(f"\n{i}. Line {pred['line_num']}\n")
            f.write(f"   Length: {pred['text_length']:,} chars\n")
            f.write(f"   Query: {pred['query']}...\n")

if __name__ == "__main__":
    # Example usage
    jsonl_file = "/home/charlie/CREST/rebuttal/failedcase/SteeringTrue_numb128_coef-1_modeafter_o_proj_correct_0_100/mix_others_low_rank_1000/Qwen/Qwen3-4B-Thinking-2507/gpqa/seed42/predictions.jsonl"
    
    if os.path.exists(jsonl_file):
        predictions, sorted_by_length, failures = analyze_predictions(jsonl_file)
    else:
        print(f"File not found: {jsonl_file}")
        print("Please update the path to your predictions.jsonl file")

