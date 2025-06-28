# Spatial-Temporal Reasoning Model

​**Jan 2024 - May 2024**​ | PyTorch, FSDP, Parallel Adapter

## Core Innovations
- ​**3D-Adapted Architecture**: Modified ViT-1.2B for spatiotemporal data
- ​**Memory-Efficient Training**: FSDP + gradient checkpointing → ​**54% ↓ mem**, ​**2× batch size**​
- ​**Adapter Fine-tuning**: 0.5% params updated → ​**​<1% acc drop**, ​**40% faster tuning**​
- ​**Optimized Inference**: Pruning + runtime optimization → ​**45% ↓ size**, ​**50% ↓ latency**​

## Applications
- Driver identification (20-class)
- Passenger status detection

---
*Contrastive pretraining adapted for dual downstream classification tasks*