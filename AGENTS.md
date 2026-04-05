# NLP301c Exam Solver - AGENTS.md

## Project Overview
- **Type**: NLP Exam Solver (Fine-tuned LLM + Vision Pipeline)
- **Path**: D:\GitHub\projects\NLP
- **Languages**: Python (primary), Shell/PowerShell for deployment
- **Frameworks**: PyTorch, Unsloth (LoRA), Transformers, Ollama (inference)
- **Models**: DeepSeek-R1-Distill-Qwen-7B (fine-tuned), qwen2.5vl:3b (vision)
- **Generated**: 2026-03-25

## Project Structure
```
NLP/
  - .sisyphus/          # Agent system plans and artifacts
  - data/               # Raw exam images and extracted JSON
    - 5 real sample/   # Exam images (SP23, SP24, SP24RE, SU24, F25)
  - docs/               # Project documentation
    - architecture/
      - decisions/      # Architecture Decision Records (ADRs)
  - predocs/            # Planning documents (original plan)
  - src/                # Source code (tokenizer, stemmer, utilities)
  - tests/              # Test suite (pytest)
  - models/             # Trained models and GGUF files (gitignored)
  - outputs/            # Training logs, checkpoints, metrics

```

## Architecture
- **Status**: Implemented - fully functional pipeline
- **Design**: Two-stage system (vision extraction → code generation)
- **Training**: LoRA on DeepSeek-R1-Distill-Qwen-7B with synthetic CoT data
- **Deployment**: Ollama on laptop CPU (offline)
- **Data**: 5 real exams + 150 synthetic samples (free OpenRouter generation)

## Key Patterns
- Chain-of-Thought (CoT) fine-tuning with <think> tags
- LoRA rank 32 for parameter-efficient tuning
- DDP multi-GPU training (2× RTX 4090)
- GGUF Q4_K_M quantization for CPU inference
- Quality validation at every stage (data, training, deployment)

## Testing Strategy
- Test Directory: tests/
- Framework: pytest
- Coverage Target: >80%
- Unit tests: tokenizer, stemmer, data validation
- Integration tests: vision extraction → code generation → execution
- End-to-end: exam image → solution file

## CI/CD
- Status: Not yet configured (GitHub Actions planned)

## Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
pytest -v
```

### Extract Real Exams from Images
```bash
python extract_exams.py
# Output: data/real_exams_raw.json (requires manual review)
# After review: real_exams.json
```

### Generate Synthetic Data (FREE - uses OpenRouter free models)
```bash
python generate_synthetic_data.py --real-exams real_exams.json --output full_exam_dataset.json
# Uses: qwen/qwen3-480b:free, deepseek/deepseek-r1:free
# Output: full_exam_dataset.json (~165 samples)
```

### Train Model

The script auto-detects GPU configuration:
- **2× RTX 4090 (48GB)**: Batch=2, Seq=2048, no 4-bit
- **2× T4 (30GB)**: Batch=1, Seq=1024, 4-bit enabled
- **Single GPU**: Batch=1, Seq=1024, 4-bit

```bash
# Quick sanity check (10 samples, 1 epoch):
python train_multigpu.py --quick-test

# Full training:
torchrun --nproc_per_node=2 train_multigpu.py

# Output: exam_r1_7b_q4/*.gguf (~4.4GB)
```

### Deploy on Laptop
```bash
# 1. Transfer GGUF files to laptop
# 2. Create Ollama model:
ollama create exam-r1-solver -f Modelfile
ollama pull qwen2.5vl:3b

# 3. Run pipeline:
python exam_solver_pipeline.py image1.jpg image2.jpg
```

## Memory Notes
- **Free API Strategy**: Uses OpenRouter free models for synthetic data (qwen3-480b:free, deepseek-r1:free) - $0 cost
- **Vision Model**: nvidia/nemotron-nano-12b-v2-vl:free for image OCR (free)
- **Training Cost**: Only electricity/cloud GPU rental (~$2-5 for 2×4090, 30 min)
- **Inference**: 100% offline, no API costs after training
- **Data**: 5 real exams extracted from WebP images → verified JSON → synthetic augmentation

## Last Updated
2026-03-25
