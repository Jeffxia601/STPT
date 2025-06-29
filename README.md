# Spatial-Temporal Reasoning Model

​**Jan 2024 - May 2024**​ | PyTorch, FSDP, Parallel Adapter | Python 3.11.5

## Core Innovations
- ​**3D-Adapted Architecture**: Modified ViT-1.2B for spatiotemporal data
- ​**Memory-Efficient Training**: FSDP + gradient checkpointing → ​**54% ↓ mem**, ​**2× batch size**​
- ​**Adapter Fine-tuning**: 0.5% params updated → ​**​<1% acc drop**, ​**40% faster tuning**​
- ​**Optimized Inference**: Pruning + runtime optimization → ​**45% ↓ size**, ​**50% ↓ latency**​

## Downstream Classification Tasks
- Driver identification (20-class)
- Passenger status detection

## 🔧 Installation
```bash
# Clone with Python 3.11+ required
git clone https://github.com/Jeffxia601/spatial-temporal-reasoning-model.git
cd spatial-temporal-reasoning-model

# Run pretraining
cd pretrain
python main.py

# Run seek-serve fine-tuning
cd ../finetune/seek_serve
python train_ss.py

# Run classification fine-tuning
cd ../classification
python train_cl.py