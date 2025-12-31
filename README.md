# CREST: Cognitive REasoning Steering at Test‑time

## TL; DR
CREST is a training-free test-time steering framework that discovers cognitive heads via simple offline calibration and then rotates activations during decoding to guide the model’s reasoning—preserving norms to avoid per-model hyperparameter tuning. This improves accuracy and reduces tokens across models and datasets.

## What is CREST?

CREST (**C**ognitive **RE**asoning **S**teering at **T**est-time) identifies attention heads whose activations are predictive of different reasoning modes (“cognitive heads”), then steers those heads at inference to suppress inefficient trajectories and encourage effective reasoning—without further training.

* Token savings with accuracy gains. e.g., R1-7B on MATH500: 92.4% with 34% fewer tokens; R1-1.5B on AMC23: 37.6% token reduction at higher accuracy. 
* Generalizes across models/datasets (DeepSeek-R1 1.5B/7B/32B, Qwen3-4B/30B, GPT-OSS-20B; MATH500, AIME, AMC23, GPQA-D, LiveCodeBench, Calendar Planning). 
* Head ratio “gold default.” Steering about the top ~38% heads (ranked by linear-probe accuracy) balances accuracy and token reduction; adopted as the default.

## Usage

### Install

Recommend use vllm docker:
```vllm/vllm-openai@sha256:d731ee65c044ae0977421eed3d93f931d4b7d79614394184c939db35b8f28fc2``` & ```docker_info.txt```

inside the docker:
```
cd CREST/probing/omni_math_rule/evaluation
pip install evaluate
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
cd ../../
pip install lighteval
pip install datasets==3.5.0
pip install emoji
```

otherwise:
```bash
cd CREST
pip install requirements.txt
cd probing/omni_math_rule/evaluation
pip install evaluate
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
cd ../../
pip install lighteval
pip install datasets==3.5.0
pip install emoji
```

### Running Baseline Experiments

```bash
# Run baseline without steering
bash script/baseline.sh
```

This script:
- Sets `STEERING=False`
- Evaluates models across multiple datasets
- Saves results in `results/SteeringFalse/` directory

### Running Steering Experiments

```bash
# Run with steering enabled
bash script/ours.sh
```

This script:
- Sets `STEERING=True`
- Configures steering parameters:
  - `steering_number=512`: Number of top steering vectors to use
  - `steering_coef=-4`: Steering coefficient (negative for inhibition)
  - `steering_mode=after_o_proj_norm_threshold`: Steering application mode
- Saves results in `results/SteeringTrue_numb512_coef-4_mode[mode]/` directory
- **NOW STEERING MUST SET ```--enforce_eager```, CUDA GRAPH AND TORCH COMPILE ARE NOT SUPPORTED YET**

#### Steering Vector Zoo
The Steering Vector Zoo contains pre-trained steering vectors that can be applied to modify model behavior during inference. These vectors are learned through probing techniques and stored as PyTorch (.pt) files.

```
probing/results/
├── [DATASET]/ # Training dataset (e.g., MATH_train)
│ └── [MODEL]/ # Model name (e.g., Qwen3-30B-A3B-Thinking-2507)
│ └── template-t0-n1-[SIZE]/ # Template and size configuration
│  └── hidden[MODE]/ # Steering mode directory
|   └── mix_others_low_rank_1000/ # Training methods
│    └── probe_best.pt # Steering vector file
```
The system automatically selects the top-k most accurate steering vectors based on:
1. **Probe Accuracy**: Classification performance on the validation set
2. **Layer-Head Coverage**: Distribution across different attention layers and heads
3. **Steering Coefficient**: Magnitude of influence (configurable via `--steering_coef`)

#### Manual Execution

```bash
python main_vllm.py \
    --model_name_or_path "Qwen/Qwen3-30B-A3B-Thinking-2507" \
    --dataset "aime25" \
    --save_dir "results/test/" \
    --use_chat_format \
    --temperature 0.6 \
    --max_tokens 32768 \
    --steering True \
    --steering_vector_path "/path/to/steering/vectors/" \
    --steering_number 512 \
    --steering_coef -4 \
    --steering_mode "after_o_proj_norm_threshold"
```

#### Required Environment Variables (for steering)

```bash
export STEERING=True
export STEERING_VECTOR_PATH="/path/to/steering/vectors/"
export STEERING_NUMBER=512
export STEERING_COEF=-4
export STEERING_MODE="after_o_proj_norm_threshold"
export MODEL_NAME_OR_PATH="Qwen/Qwen3-30B-A3B-Thinking-2507"
```

## Repository overview

The implementation consists of three main components:

### 1. Main Inference Engine (`main_vllm.py`)
- **Purpose**: Core inference script supporting multiple datasets and models
- **Features**:
  - Support for multiple datasets (MATH, GSM, AIME, GPQA, LiveCodeBench, etc.)
  - Batch processing with vLLM backend
  - Configurable steering parameters
  - Multi-model support with automatic model detection

### 2. Steering Implementation (`probing/modeling_utils/vllm/`)
Contains model-specific monkey patches for different architectures:

- **`qwen2/monkey_patch.py`**: DeepSeek-R1-Distill-Qwen models
- **`qwen3/monkey_patch.py`**: Qwen3-4B-Thinking models  
- **`qwen3_moe/monkey_patch.py`**: Qwen3-30B-A3B-Thinking models
- **`gpt_oss/monkey_patch.py`**: GPT-OSS models

Each monkey patch implements:
- Attention mechanism modifications
- Layer-wise steering vector application
- Dynamic steering flag management
- Multiple steering modes

### 3. Evaluation System (`probing/get_omni_results.py`)
- **Purpose**: Mathematical reasoning evaluation using OmniMath rules
- **Features**:
  - Parallel evaluation with timeout handling
  - Multiple prediction aggregation
  - Accuracy computation and result saving

## Steering Modes

The system supports four distinct steering modes, each applying steering vectors at different points in the attention mechanism:

### 1. `before_o_proj`
- **Application Point**: Before the output projection in attention
- **Mechanism**: Modifies attention output before linear transformation
- **Use Case**: Early intervention in attention computation

### 2. `after_o_proj` 
- **Application Point**: After the output projection in attention
- **Mechanism**: Direct modification of projected attention output
- **Use Case**: Standard steering application

### 3. `after_o_proj_norm`
- **Application Point**: After output projection with norm preservation
- **Mechanism**: Applies steering while preserving vector norms
- **Use Case**: Maintaining activation magnitude consistency

### 4. `after_o_proj_norm_threshold`
- **Application Point**: After output projection with thresholded norm preservation
- **Mechanism**: Applies steering with awareness thresholding and norm preservation
- **Use Case**: Selective steering based on activation patterns

## Supported Datasets

- **"aime25"**
- **"AIME"**
- **"amc23"**
- **"cp"**
- **"lcb"**
- **"MATH500"**
- **"GSM"**
- **"gpqa"**

## Supported Models

The system automatically detects and applies appropriate steering based on model names:

- **DeepSeek-R1-Distill-Qwen**: Uses `qwen2` monkey patch
- **Qwen3-4B**: Uses `qwen3` monkey patch  
- **Qwen3-30B**: Uses `qwen3_moe` monkey patch
- **gpt-oss**: Uses `gpt_oss` monkey patch


## Steering Vector Format

Steering vectors should be saved as PyTorch files with the following structure:

```python
{
    layer_idx: {
        head_idx: {
            'accuracy': float,           # Probe accuracy
            'model_dict': {
                'weight': torch.Tensor   # Hyperplane weights
            },
            'steering_vector': torch.Tensor  # Steering direction
        }
    }
}
```

## Output Structure

Results are saved in the following structure:
```
results/
├── SteeringFalse/                    # Baseline results
│   └── [model]/[dataset]/seed[X]/
├── SteeringTrue_numb[N]_coef[C]_mode[M]/  # Steering results
│   └── [model]/[dataset]/
└── predictions.jsonl                 # Detailed predictions
└── metrics.json                      # Evaluation metrics
```

## Key Features

- **Dynamic Steering**: Steering is applied based on token patterns (e.g., double newlines)
- **Multi-GPU Support**: Tensor parallelism with proper rank handling
- **Batch Processing**: Efficient batch inference with vLLM
- **Comprehensive Evaluation**: Multiple evaluation metrics per dataset
- **Flexible Configuration**: Easy parameter tuning through environment variables

## Technical Implementation

### Steering Mechanism

1. **Vector Loading**: Top-k steering vectors are loaded based on probe accuracy
2. **Flag Detection**: Steering flags are set based on specific token patterns
3. **Awareness Computation**: Hyperplane-based awareness scores determine steering strength
4. **Vector Application**: Steering vectors are applied with configurable coefficients
5. **Norm Preservation**: Optional norm preservation maintains activation magnitudes

### Model Patching

The system uses runtime monkey patching to modify:
- `forward()` methods in attention layers
- `forward()` methods in decoder layers  
- `forward()` methods in the main model
- Token-based steering flag management

## Performance Considerations

- **Memory**: Steering vectors are loaded onto GPU memory
- **Scalability**: Supports tensor parallelism for large models
- **Efficiency**: Conditional application reduces unnecessary computation

## Research Applications

This implementation supports research in:
- Mechanistic interpretability of reasoning
- Activation steering and control
- Cognitive enhancement in language models
- Mathematical reasoning improvement
- Code generation optimization

## Citation

If you use this code or ideas, please cite the CREST paper:

*Understanding and Steering The Cognitive Behaviors of Reasoning Models At Test-Time (CREST)*