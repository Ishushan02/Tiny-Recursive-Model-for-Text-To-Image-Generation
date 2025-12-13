# TRM: Tiny Recursive Multimodal Text-to-Image Generation

This repository implements **TRM (Tiny Recursive Model)** — a research oriented **text-to-image generation framework** that combines **multi-model text embeddings**, **DCAE image latents**, and a **recursive transformer-style reasoning loop** for image synthesis.

The model is trained on **COCO-style image–caption datasets** and learns to denoise images conditioned on rich text embeddings through **iterative latent reasoning**.

---

## Architecture Overview

![Architecture](architecture.jpg)

The system consists of:
- Multi-model text embedding fusion
- DCAE-based image latent encoding/decoding
- Patch-level image–text fusion
- Recursive transformer reasoning blocks
- Pixel and perceptual loss supervision

---

## Key Ideas

- Text embeddings are **concatenated from multiple pretrained models**
- Images are encoded into compact **DCAE latent representations**
- Image and text tokens are fused into a shared embedding space
- A **recursive transformer block** refines latents iteratively
- Training uses **multi-step supervision** with perceptual guidance

---

## Text Embedding Pipeline

Text embeddings are formed by concatenating hidden states from:

- `BAAI/bge-base-en-v1.5`
- `nomic-ai/nomic-embed-text-v1`
- `Qwen/Qwen3-Embedding-8B`

Final text embedding shape:
```
(batch, 512, 5632)
```

---

## Image Encoding

- Images are resized to **512×512**
- Encoded using **MIT-HAN DCAE**
- Latent representation:
```
(batch, 128, 8, 8)
```

Decoding back to RGB space is performed using the same pretrained autoencoder.

---

## Image–Text Fusion

### ConcatenateImgTextMLP

- Image latents are patchified and positionally encoded
- Text embeddings are projected to a 768-dimensional space
- Modality embeddings distinguish image and text tokens
- Output is a unified multimodal token sequence

---

## Recursive Transformer Reasoning

### TinyRecursiveBlock

The core reasoning module performs:

1. Time-conditioned adaptive layer normalization
2. Multi-head self-attention
3. Feedforward transformation
4. Recursive latent refinement

Two latent states are maintained:
- **y** — image latent state
- **z** — reasoning latent state

The model performs deep reasoning without gradients, followed by a final gradient-tracked step.

---

## Output Head

- Projects refined latents back to spatial image latents
- Reconstructs full-resolution images using DCAE decoding

Output shape:
```
(batch, 3, 512, 512)
```

---

## Training Objective

Total loss:
```
Total Loss = MSE Loss + 0.25 × Perceptual Loss
```

- **MSE Loss** enforces pixel-level accuracy
- **Perceptual Loss** compares VGG16 feature activations

---

## Training Details

| Parameter | Value |
|--------|------|
| Batch Size | 2 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 3e-2 |
| Scheduler | StepLR |
| Supervision Steps | 16 |
| Attention Heads | 16 |
| Embedding Dim | 768 |

---

## Dataset Format

Dataset CSV must contain:
```csv
imagePath,caption1,caption2,caption3,caption4,caption5
```

Images are loaded relative to the dataset root directory.

---

## Checkpointing

Checkpoints are saved to:
```
models/trmModel.pt
```

Each checkpoint includes:
- Model state
- Optimizer state
- Scheduler state
- Epoch index

Training resumes automatically if a checkpoint exists.

---

## Experiment Tracking

- Integrated with **Weights & Biases**
- Logs training loss and learning rate

---

## Running Training

```bash
python train.py
```

Ensure CUDA is available and dataset paths are correctly configured.

---

## Notes

- Designed for research experimentation
- Memory intensive due to large text embeddings
- Recursive reasoning improves stability over deep single-pass transformers
- Easily extendable to diffusion-style conditioning

---

## License

For research and educational use only.
