#!/usr/bin/env python3
"""
Fine-tuning script for Athena philosophical model.
Based on Qwen 3.5 9B Abliterated (de-censored, heretic version).
Optimized for RTX 4090 24GB with 4-bit QLoRA.
"""

import os
import sys
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json
import logging
from datetime import datetime
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_athena.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(jsonl_path):
    """Load Athena dataset from JSONL file."""
    logger.info(f"Loading dataset from {jsonl_path}")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Convert to HuggingFace Dataset
    prompts = [item.get('prompt', '') for item in data]
    completions = [item.get('completion', '') for item in data]
    
    # Combine prompt + completion for causal LM training
    texts = []
    for prompt, completion in zip(prompts, completions):
        # Format for instruction fine-tuning
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{completion}\n\n### End"
        texts.append(text)
    
    dataset = Dataset.from_dict({"text": texts})
    
    # Stats
    total_chars = sum(len(t) for t in texts)
    total_tokens_est = total_chars // 4  # Rough estimate
    logger.info(f"Dataset stats: {len(texts)} examples, ~{total_tokens_est:,} tokens")
    logger.info(f"Avg tokens per example: ~{total_tokens_est // len(texts):,}")
    
    return dataset

def tokenize_dataset(dataset, tokenizer):
    """Tokenize dataset with proper padding/truncation."""
    logger.info("Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Tokenize without padding (we'll pad later in collator)
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,  # Conservative for 9B model
            return_tensors=None,
            add_special_tokens=True
        )
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    # Calculate token counts
    total_tokens = sum(len(seq) for seq in tokenized_dataset["input_ids"])
    logger.info(f"Total tokens after tokenization: {total_tokens:,}")
    
    return tokenized_dataset

def setup_model_and_tokenizer(model_name):
    """Setup 4-bit quantized model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    # 4-bit quantization config for RTX 4090 efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Double quantization for extra savings
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Auto device placement
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False  # Disable cache for gradient checkpointing
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # Alpha parameter
        lora_dropout=0.1,
        bias="none",
        target_modules=[  # Target modules for Qwen architecture
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        modules_to_save=["embed_tokens", "lm_head"]  # Keep these trainable
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}% of {total_params:,})")
    
    return model, tokenizer

def main():
    """Main training function."""
    # Configuration
    CONFIG = {
        "model_name": "lukey03/Qwen3.5-9B-abliterated",
        "dataset_path": "/workspace/datasets/athena/athena_training_completo.jsonl",
        "output_dir": "/workspace/models/athena_finetuned",
        "run_name": f"athena_qwen35_9b_{datetime.now().strftime('%Y%m%d_%H%M')}",
        
        # Training hyperparameters (optimized for 9B on RTX 4090)
        "num_train_epochs": 4,
        "per_device_train_batch_size": 2,  # Small due to 9B model
        "gradient_accumulation_steps": 8,   # Effective batch size = 16
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_grad_norm": 0.3,
        
        # Optimization
        "optim": "paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
        "lr_scheduler_type": "cosine",
        "fp16": True,
        
        # Checkpointing
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 3,
        "logging_steps": 50,
        "eval_steps": 500,
        
        # Early stopping
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "early_stopping_patience": 3,
    }
    
    logger.info("=" * 60)
    logger.info("ATHENA PHILOSOPHICAL MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model: {CONFIG['model_name']}")
    logger.info(f"Output: {CONFIG['output_dir']}")
    logger.info(f"Run: {CONFIG['run_name']}")
    
    # Initialize wandb (optional)
    try:
        wandb.init(
            project="athena-philosophical",
            name=CONFIG["run_name"],
            config=CONFIG
        )
        logger.info("WandB initialized")
    except:
        logger.warning("WandB not available, continuing without")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(CONFIG["model_name"])
    
    # Load dataset
    dataset = load_dataset(CONFIG["dataset_path"])
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Split dataset (90% train, 10% eval)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Eval examples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        run_name=CONFIG["run_name"],
        
        # Training loop
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        
        # Optimization
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_steps=CONFIG["warmup_steps"],
        max_grad_norm=CONFIG["max_grad_norm"],
        
        # Scheduler
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        
        # Optimizer
        optim=CONFIG["optim"],
        
        # Precision
        fp16=CONFIG["fp16"],
        
        # Checkpointing
        save_strategy=CONFIG["save_strategy"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        
        # Early stopping
        load_best_model_at_end=CONFIG["load_best_model_at_end"],
        metric_for_best_model=CONFIG["metric_for_best_model"],
        greater_is_better=CONFIG["greater_is_better"],
        
        # Other
        report_to="wandb" if wandb.run else "none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"])]
    )
    
    # Train!
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Save training config
    with open(os.path.join(CONFIG["output_dir"], "training_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {CONFIG['output_dir']}")
    logger.info("=" * 60)
    
    # Print VRAM usage summary
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"VRAM usage: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())