#!/usr/bin/env python3
"""
Setup script for Athena training on RunPod.
Prepares environment, installs dependencies, and uploads files.
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path

def check_environment():
    """Check if we're in a RunPod environment."""
    print("🔍 Checking environment...")
    
    # Check for RunPod specific env vars
    runpod_env_vars = ["RUNPOD_POD_ID", "RUNPOD_API_KEY", "RUNPOD_GPU_COUNT"]
    found = any(os.getenv(var) for var in runpod_env_vars)
    
    if found:
        print("✅ Running in RunPod environment")
        pod_id = os.getenv("RUNPOD_POD_ID", "unknown")
        print(f"   Pod ID: {pod_id}")
        return True
    else:
        print("⚠️  Not in RunPod environment (local testing)")
        return False

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    requirements = [
        "torch==2.3.0",
        "transformers==4.40.0",
        "accelerate==0.28.0",
        "peft==0.10.0",
        "datasets==2.19.0",
        "bitsandbytes==0.43.0",
        "wandb==0.17.0",
        "scipy==1.13.0",
        "pyyaml==6.0.1",
        "huggingface-hub==0.22.2",
    ]
    
    # Write requirements file
    with open("requirements_athena.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    try:
        # Install with pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade"] + requirements,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during installation: {e}")
        return False
    
    return True

def setup_workspace():
    """Create workspace directory structure."""
    print("\n📁 Setting up workspace...")
    
    directories = [
        "/workspace/datasets/athena",
        "/workspace/models/athena_finetuned",
        "/workspace/logs",
        "/workspace/checkpoints",
        "/workspace/config",
        "/workspace/scripts",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    return True

def verify_dataset():
    """Verify Athena dataset exists and is valid."""
    print("\n📊 Verifying dataset...")
    
    dataset_path = "/workspace/datasets/athena/athena_training_completo.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Count examples
    count = 0
    total_chars = 0
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                count += 1
                total_chars += len(line)
        
        estimated_tokens = total_chars // 4
        
        print(f"✅ Dataset found: {dataset_path}")
        print(f"   Examples: {count}")
        print(f"   Estimated tokens: {estimated_tokens:,}")
        print(f"   File size: {os.path.getsize(dataset_path) / (1024*1024):.1f} MB")
        
        if count < 100:
            print("⚠️  Warning: Dataset has fewer than 100 examples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def load_config():
    """Load training configuration."""
    print("\n⚙️  Loading configuration...")
    
    config_path = "/workspace/config/training-athena.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded: {config_path}")
        print(f"   Model: {config.get('model', {}).get('name', 'N/A')}")
        print(f"   Dataset: {config.get('training', {}).get('estimated_tokens', 0):,} tokens")
        print(f"   Epochs: {config.get('training', {}).get('epochs', 0)}")
        
        return config
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def estimate_training_time(config):
    """Estimate training time based on config."""
    if not config:
        return
    
    print("\n⏱️  Estimating training time...")
    
    training = config.get('training', {})
    hardware = config.get('hardware', {})
    
    estimated_tokens = training.get('estimated_tokens', 400000)
    batch_size = training.get('batch_size', 2)
    gradient_accumulation = training.get('gradient_accumulation', 8)
    epochs = training.get('epochs', 4)
    
    # Effective batch size
    effective_batch = batch_size * gradient_accumulation
    
    # Tokens per step
    tokens_per_step = effective_batch * 2048  # Assuming max_length=2048
    
    # Steps per epoch
    steps_per_epoch = estimated_tokens // tokens_per_step
    
    # Total steps
    total_steps = steps_per_epoch * epochs
    
    # Time estimation (conservative)
    seconds_per_step = 2.5  # Conservative for 9B on RTX 4090
    total_seconds = total_steps * seconds_per_step
    total_hours = total_seconds / 3600
    
    # Cost estimation
    cost_per_hour = hardware.get('cost_per_hour', 0.59)
    estimated_cost = total_hours * cost_per_hour
    
    print(f"   Estimated tokens: {estimated_tokens:,}")
    print(f"   Effective batch size: {effective_batch}")
    print(f"   Estimated steps: {total_steps:,}")
    print(f"   Estimated time: {total_hours:.1f} hours")
    print(f"   Estimated cost: ${estimated_cost:.2f} (~€{estimated_cost * 0.92:.2f})")
    
    return total_hours, estimated_cost

def check_gpu():
    """Check GPU availability and specs."""
    print("\n🎮 Checking GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available, GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                print(f"   GPU {i}: {gpu_name}")
                print(f"      Memory: {memory_total:.1f} GB total")
                print(f"      Currently: {memory_allocated:.1f} GB allocated, {memory_reserved:.1f} GB reserved")
                
                if memory_total < 20:
                    print(f"   ⚠️  Warning: GPU has less than 20GB VRAM")
                    
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("ATHENA TRAINING SETUP")
    print("=" * 60)
    
    # Check environment
    is_runpod = check_environment()
    
    # Setup workspace
    if not setup_workspace():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing despite dependency issues...")
    
    # Verify dataset
    if not verify_dataset():
        print("❌ Dataset verification failed")
        return 1
    
    # Load config
    config = load_config()
    if not config:
        print("❌ Config loading failed")
        return 1
    
    # Estimate training time
    estimate_training_time(config)
    
    # Check GPU
    if not check_gpu():
        print("⚠️  GPU check failed, but continuing...")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n🎯 Next steps:")
    print("1. Start training: python /workspace/scripts/train_athena.py")
    print("2. Monitor logs: tail -f /workspace/logs/training_athena.log")
    print("3. Check progress on WandB (if configured)")
    
    if is_runpod:
        print("\n⚠️  RunPod Notes:")
        print("   - Training will auto-terminate after estimated time + buffer")
        print("   - Check cost regularly on RunPod console")
        print("   - Save checkpoints frequently")
    
    # Create a ready flag file
    with open("/workspace/READY_FOR_TRAINING", "w") as f:
        f.write(f"Athena training setup complete at {subprocess.check_output(['date']).decode().strip()}\n")
    
    print("\n✅ Ready for Athena philosophical training!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())