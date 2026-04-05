#!/usr/bin/env python3
"""
Multi-GPU Training Script for NLP Exam Solver
===============================================

Fine-tunes DeepSeek-R1-Distill-Qwen-7B using LoRA on synthetic exam data.
Supports DDP (Distributed Data Parallel) for 2× RTX 4090 setup.

Usage:
    # Full training on 2 GPUs:
    torchrun --nproc_per_node=2 train_multigpu.py

    # Quick sanity check (10 samples, 1 epoch):
    python train_multigpu.py --quick-test

    # Single GPU:
    python train_multigpu.py --num-gpus 1
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

# Hyperparameters (auto-adjusted based on GPU count and VRAM)
NUM_GPUS = torch.cuda.device_count()

# Detect GPU type and adjust accordingly
GPU_NAME = torch.cuda.get_device_name(0) if NUM_GPUS > 0 else "unknown"
IS_T4 = "T4" in GPU_NAME or "Tesla" in GPU_NAME
TOTAL_VRAM_GB = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(NUM_GPUS))

print(f"🚀 Detected {NUM_GPUS} GPU(s): {GPU_NAME}")
print(f"   Total VRAM: {TOTAL_VRAM_GB:.1f} GB")

# Adjust hyperparameters based on available VRAM
if TOTAL_VRAM_GB >= 40:  # 2x RTX 4090 or equivalent
    BATCH_SIZE_PER_GPU = 2
    GRADIENT_ACCUMULATION = 4
    MAX_SEQ_LENGTH = 2048
    USE_4BIT = False
elif TOTAL_VRAM_GB >= 25:  # 2x T4 (30GB) or similar
    BATCH_SIZE_PER_GPU = 1
    GRADIENT_ACCUMULATION = 8
    MAX_SEQ_LENGTH = 1024  # Reduced for T4
    USE_4BIT = True  # Use 4-bit quantization to save memory
else:  # Single GPU or less VRAM
    BATCH_SIZE_PER_GPU = 1
    GRADIENT_ACCUMULATION = 8
    MAX_SEQ_LENGTH = 1024
    USE_4BIT = True

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
WARMUP_RATIO = 0.1

# Training output
OUTPUT_DIR = "outputs_nlp_exam"
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"


def print_config():
    """Print training configuration."""
    print("\n" + "="*60)
    print("  TRAINING CONFIGURATION")
    print("="*60)
    print(f"  Model:          {MODEL_NAME}")
    print(f"  GPU:            {GPU_NAME}")
    print(f"  Total VRAM:     {TOTAL_VRAM_GB:.1f} GB")
    print(f"  4-bit Quant:    {USE_4BIT}")
    print(f"  GPUs:           {NUM_GPUS}")
    print(f"  Batch/GPU:      {BATCH_SIZE_PER_GPU}")
    print(f"  Grad Accum:     {GRADIENT_ACCUMULATION}")
    print(f"  Effective B:    {BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION * NUM_GPUS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  LoRA Rank:      {LORA_R}")
    print(f"  Max Seq Len:    {MAX_SEQ_LENGTH}")
    print(f"  Output Dir:     {OUTPUT_DIR}")
    print("="*60 + "\n")


def load_dataset(dataset_path: str):
    """Load and prepare training dataset."""
    from datasets import Dataset
    
    print(f"📂 Loading dataset: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both formats: {"metadata": ..., "data": [...]} or direct list
    if isinstance(data, dict) and 'data' in data:
        samples = data['data']
        print(f"   Metadata: {data.get('metadata', {})}")
    else:
        samples = data
    
    # Convert to training format
    def format_sample(sample):
        """Convert sample to R1 chain-of-thought format."""
        user_msg = sample.get('instruction', '')
        
        # Add input if present
        if sample.get('input') and sample.get('input') != 'None':
            user_msg += f"\n\nInput: {sample['input']}"
        
        # Add expected output if present
        if sample.get('expected_output') and sample.get('expected_output') != 'None':
            user_msg += f"\n\nExpected output: {sample['expected_output']}"
        
        # Use provided thinking or generate simple one
        thinking = sample.get('thinking', 
            f"Let me think through this problem step by step:\n"
            f"1. The task requires: {sample.get('instruction', '')[:80]}...\n"
            f"2. I'll need to use appropriate NLTK functions\n"
            f"3. Let me write clean, executable Python code")
        
        # R1 format with <think> tags
        r1_text = f"""<｜begin▁of▁sentence｜><｜User｜>{user_msg}<｜Assistant｜><｜think｜>
{thinking}
<｜/think｜>

{sample.get('output', '')}<｜end▁of▁sentence｜>"""
        
        return {"text": r1_text}
    
    # Format all samples
    formatted = [format_sample(s) for s in samples if 'instruction' in s and 'output' in s]
    
    print(f"   Loaded {len(formatted)} valid samples")
    
    # Split into train/eval (90/10)
    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"   Train: {len(split['train'])} samples")
    print(f"   Eval:  {len(split['test'])} samples")
    
    return split


def setup_model_tokenizer():
    """Load model and tokenizer with LoRA configuration."""
    from unsloth import FastLanguageModel
    
    print(f"🔄 Loading model: {MODEL_NAME}")
    
    # Load with LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (bf16 if available)
        load_in_4bit=True,  # Use 4-bit quantization to save VRAM
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    
    print(f"   ✅ Model loaded with LoRA (rank={LORA_R})")
    
    return model, tokenizer


def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Run the training loop."""
    from transformers import TrainingArguments, Trainer
    from unsloth import is_bfloat16_supported
    
    # Calculate total steps
    total_samples = len(train_dataset)
    effective_batch = BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION * NUM_GPUS
    steps_per_epoch = total_samples // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    print(f"📊 Training stats:")
    print(f"   Total samples:  {total_samples}")
    print(f"   Effective batch: {effective_batch}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps:     {total_steps}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        per_device_eval_batch_size=BATCH_SIZE_PER_GPU,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=max(10, steps_per_epoch // 3),  # At least 3 evals per epoch
        save_strategy="steps",
        save_steps=max(20, steps_per_epoch // 2),
        save_total_limit=2,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=["tensorboard"] if os.getenv("WANDB_API_KEY") else [],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # Important for DDP
        dataloader_num_workers=4,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("\n🚀 Starting training...")
    print("   (Use 'tensorboard --logdir outputs_nlp_exam' to monitor)\n")
    
    # Train
    train_result = trainer.train()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("\n✅ Training complete!")
    print(f"   Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    
    return trainer


def export_to_gguf(trainer, tokenizer, output_dir="exam_r1_7b_q4"):
    """Export trained model to GGUF format for Ollama."""
    print(f"\n📦 Exporting to GGUF format...")
    
    # Save adapter first
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Export to GGUF using Unsloth
    try:
        trainer.model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"   ✅ GGUF exported to: {output_dir}/")
    except Exception as e:
        print(f"   ⚠️  GGUF export failed: {e}")
        print("   💡 Alternative: Use llama.cpp to convert manually:")
        print(f"      llama.cpp convert.py {output_dir} --outfile exam_r1_7b_q4.gguf --quantize q4_k_m")
    
    return output_dir


def quick_sanity_check(model, tokenizer, dataset_path: str):
    """Quick test: train on 10 samples for 1 epoch to verify pipeline works."""
    print("\n🧪 QUICK SANITY CHECK (10 samples, 1 epoch)")
    print("="*50)
    
    # Load only first 10 samples
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('data', data)[:10]
    
    def format_sample(sample):
        user_msg = sample.get('instruction', '')
        if sample.get('input'):
            user_msg += f"\n\nInput: {sample['input']}"
        
        thinking = sample.get('thinking', "I'll solve this step by step.")
        
        r1_text = f"""<｜begin▁of▁sentence｜><｜User｜>{user_msg}<｜Assistant｜><｜think｜>
{thinking}
<｜/think｜>

{sample.get('output', '')}<｜end▁of▁sentence｜>"""
        
        return {"text": r1_text}
    
    from datasets import Dataset
    formatted = [format_sample(s) for s in samples if 'instruction' in s and 'output' in s]
    dataset = Dataset.from_list(formatted)
    
    print(f"   Dataset: {len(dataset)} samples")
    
    # Load model
    model, tokenizer = setup_model_tokenizer()
    
    # Quick training
    from transformers import TrainingArguments, Trainer
    from unsloth import is_bfloat16_supported
    
    training_args = TrainingArguments(
        output_dir="outputs_quick_test",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=LEARNING_RATE,
        bf16=is_bfloat16_supported(),
        logging_steps=2,
        save_strategy="no",
        report_to=[],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("   Training...")
    trainer.train()
    
    # Check loss decreased
    final_loss = trainer.state.log_history[-1].get('loss', None)
    if final_loss and final_loss < 2.0:
        print(f"   ✅ Sanity check PASSED (loss: {final_loss:.4f})")
        return True
    else:
        print(f"   ⚠️  Loss: {final_loss}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train NLP exam solver model")
    parser.add_argument("--dataset", type=str, default="full_exam_dataset.json",
                        help="Path to training dataset")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Force number of GPUs (auto-detect if not specified)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick sanity check with 10 samples")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_PER_GPU,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=LORA_R,
                        help="LoRA rank")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip GGUF export")
    
    args = parser.parse_args()
    
    global NUM_GPUS, BATCH_SIZE_PER_GPU, GRADIENT_ACCUMULATION, NUM_EPOCHS
    
    if args.num_gpus:
        NUM_GPUS = args.num_gpus
        BATCH_SIZE_PER_GPU = 2 if NUM_GPUS >= 2 else 4
        GRADIENT_ACCUMULATION = 4 if NUM_GPUS >= 2 else 8
    
    NUM_EPOCHS = args.epochs
    BATCH_SIZE_PER_GPU = args.batch_size
    
    print_config()
    
    # Quick test mode
    if args.quick_test:
        success = quick_sanity_check(None, None, args.dataset)
        if success:
            print("\n✅ Pipeline verified! Ready for full training.")
        else:
            print("\n❌ Pipeline check failed. Check data format and model.")
        return
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print("   Generate with: python generate_synthetic_data.py")
        sys.exit(1)
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Setup model
    model, tokenizer = setup_model_tokenizer()
    
    # Train
    trainer = train_model(model, tokenizer, dataset['train'], dataset['test'])
    
    # Export to GGUF
    if not args.no_export:
        export_to_gguf(trainer, tokenizer)
    
    print("\n🎉 All done!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print("   Next: Transfer to laptop and create Ollama model")


if __name__ == "__main__":
    main()
