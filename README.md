# Athena AI - Philosophical Model Training

Fine-tuning of Qwen 3.5 9B Abliterated (de-censored, heretic version) for Athena philosophical AI.

## 🎯 Quick Start (RunPod)

One-command deployment:

```bash
curl -s https://raw.githubusercontent.com/anzugtp-dot/athena-ai/main/deploy.sh | bash
```

## 📊 Specifications

- **Base Model**: Qwen 3.5 9B Abliterated (de-censored, heretic)
- **Dataset**: 320 philosophical examples (~400k tokens)
- **Training**: 4-bit QLoRA on RTX 4090
- **Time**: 12-16 hours
- **Cost**: €7.58-€9.21

## 📁 Files

- `datasets/athena_training_completo.jsonl` - 320 philosophical examples
- `scripts/train_athena.py` - Main training script
- `scripts/setup_athena_training.py` - Environment setup
- `config/training-athena.yaml` - Training configuration
- `deploy.sh` - One-click deployer

## 🚀 Training

```bash
# On RunPod pod:
cd /workspace
python scripts/setup_athena_training.py
python scripts/train_athena.py
```

## 📈 Monitoring

- Logs: `/workspace/logs/training_athena.log`
- Checkpoints: `/workspace/checkpoints/`
- Models: `/workspace/models/athena_finetuned/`

## 🔧 Requirements

See `requirements.txt` for Python dependencies.

## 📞 Support

- GitHub: https://github.com/anzugtp-dot/athena-ai
- Telegram: @Anzu777

---

**Athena Training Status**: 🟢 READY  
**Repository**: https://github.com/anzugtp-dot/athena-ai  
**Created**: 2026-04-03 14:52
