# notes2latex-ocr

A Python-based neural network for translating handwritten text into LaTeX markup.
Built with **Python + PyTorch** following the **pix2tex** approach.

## Architecture

```
Photo of a handwritten expression
        ↓
Breakdown into patches (e.g. 16×16 pixels)
        ↓
ViT Encoder  (Transformer over patches)
        ↓
Transformer Decoder  (generates LaTeX token by token)
        ↓
LaTeX string
```

### Components

| Module | File | Description |
|---|---|---|
| `PatchEmbedding` | `src/model/encoder.py` | Splits the image into non-overlapping square patches and projects each to an embedding vector via a strided convolution |
| `ViTEncoder` | `src/model/encoder.py` | Adds sinusoidal positional embeddings and passes the patch sequence through N transformer encoder blocks |
| `TransformerDecoder` | `src/model/decoder.py` | Autoregressively generates LaTeX tokens using masked self-attention + cross-attention over the encoded patches |
| `Pix2Tex` | `src/model/model.py` | End-to-end model combining the encoder and decoder; exposes `forward` (teacher-forced, for training) and `generate` (greedy, for inference) |
| `CharTokenizer` | `src/data/dataset.py` | Minimal character-level tokeniser for LaTeX strings |
| `LatexDataset` | `src/data/dataset.py` | `torch.utils.data.Dataset` that loads `(image, LaTeX)` pairs from a tab-separated manifest file |

## Installation

```bash
pip install -r requirements.txt
```

## Data format

Each entry in a manifest `.tsv` file is a tab-separated pair:

```
/absolute/path/to/image.png	\frac{1}{2} + x^2
```

Lines starting with `#` are treated as comments.

## Training

```bash
python -m src.train \
    --train_manifest data/train.tsv \
    --val_manifest   data/val.tsv   \
    --output_dir     checkpoints/   \
    --epochs 30 --batch_size 32 --lr 3e-4
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--image_height` | 128 | Input image height (px) |
| `--image_width` | 512 | Input image width (px) |
| `--patch_size` | 16 | Patch side length (px) |
| `--embed_dim` | 256 | Shared embedding dimension |
| `--encoder_depth` | 6 | Number of ViT encoder blocks |
| `--decoder_depth` | 6 | Number of decoder blocks |
| `--num_heads` | 8 | Attention heads |
| `--max_seq_len` | 512 | Maximum output token length |

## Inference

```bash
python -m src.predict \
    --checkpoint checkpoints/best_model.pt \
    --vocab_file  data/vocab.txt \
    --image       path/to/image.png
```

## Tests

```bash
python -m pytest tests/ -v
```
