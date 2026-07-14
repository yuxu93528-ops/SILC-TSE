# SILC-TSE

## Lightweight Causal Target Speaker Extraction via Staged Condition Injection and Learnable Frequency Compression

This repository provides an inference-only demo for qualitative evaluation of SILC-TSE, a lightweight causal target speaker extraction model.

Given a two-speaker speech mixture and a reference utterance from the target speaker, SILC-TSE extracts the speech signal corresponding to the referenced speaker.

This repository is not a full reproduction package. Training code, the original PyTorch implementation, experimental configurations, data preparation scripts, dataset construction scripts, and the full evaluation pipeline are not included.

## Model Interface

The demo model is provided as an ONNX model and is executed with ONNX Runtime on CPU.

| Item | Description |
|---|---|
| Mixture input | Two-speaker mixture waveform |
| Reference input | Reference waveform from the target speaker |
| Output | Estimated waveform of the target speaker |
| Sampling rate | 8 kHz |
| Duration | 4 seconds |
| Samples | 32000 samples |
| Input shape | `[1, 1, 32000]` |
| Output shape | `[1, 1, 32000]` |
| Batch size | 1 |
| Runtime provider | `CPUExecutionProvider` |

## Repository Structure

```text
SILC-TSE/
├── checkpoints/
│   └── silc_tse_demo.onnx
├── examples/
│   ├── showcase_01/
│   ├── showcase_02/
│   └── showcase_03/
├── inference.py
├── requirements.txt
└── README.md
```

## Installation

Install the minimal inference dependencies:

```bash
pip install -r requirements.txt
```

## Run Inference

Run the following commands from the repository root.

Extract speaker 1 using `reference_s1.wav`:

```bash
python inference.py --mixture examples/showcase_01/mixture.wav --reference examples/showcase_01/reference_s1.wav --model checkpoints/silc_tse_demo.onnx --output outputs/showcase_01_s1.wav
```

Extract speaker 2 using `reference_s2.wav`:

```bash
python inference.py --mixture examples/showcase_01/mixture.wav --reference examples/showcase_01/reference_s2.wav --model checkpoints/silc_tse_demo.onnx --output outputs/showcase_01_s2.wav
```

The output directory is created automatically if it does not already exist.

Input audio is converted to mono and resampled to 8 kHz. Audio shorter than 4 seconds is zero-padded, while audio longer than 4 seconds is truncated to the first 4 seconds.

## Demo Audio Files

Each showcase directory contains the following files:

| File | Description |
|---|---|
| `mixture.wav` | Input two-speaker mixture |
| `clean_s1.wav` | Clean source signal for speaker 1 |
| `clean_s2.wav` | Clean source signal for speaker 2 |
| `reference_s1.wav` | Reference utterance from speaker 1 |
| `reference_s2.wav` | Reference utterance from speaker 2 |
| `estimate_s1.wav` | SILC-TSE output using `reference_s1.wav` |
| `estimate_s2.wav` | SILC-TSE output using `reference_s2.wav` |

## Demo Examples

| Example | Files |
|---|---|
| Showcase 01 | [Open showcase_01](examples/showcase_01/) |
| Showcase 02 | [Open showcase_02](examples/showcase_02/) |
| Showcase 03 | [Open showcase_03](examples/showcase_03/) |

The three examples are curated for qualitative listening and are not intended to replace the quantitative evaluation reported in the paper.

## Datasets

For model evaluation, two target speaker extraction datasets were constructed
from the WSJ0 and AISHELL-1 speech corpora. Each TSE sample consists of a
two-speaker mixture, a clean target signal, and a reference utterance spoken
by the same target speaker.

The source utterances were converted to 8 kHz and mixed at randomly selected
relative energy levels. Each mixture was used twice by alternately treating
the two speakers as the extraction target.

| Dataset | Training | Validation | Test | Relative energy range |
|---|---:|---:|---:|---:|
| WSJ0-2mix | 20,000 | 5,000 | 3,000 | -5 dB to 5 dB |
| AISHELL-2mix | 100,000 | 5,000 | 8,000 | -6 dB to 6 dB |

WSJ0-2mix was constructed from English utterances in WSJ0, while
AISHELL-2mix was constructed from Mandarin utterances in AISHELL-1.
Utterances shorter than 2 seconds were excluded when constructing
AISHELL-2mix.

The three qualitative examples provided in this repository were selected from
the WSJ0-2mix test set. Full dataset partitions, construction scripts, and
evaluation files are not included. Further construction details are provided
in the accompanying paper.

## Demo Data

The audio examples included in this repository are three short, curated qualitative examples derived from the WSJ0-2mix test set. They are provided only for qualitative listening with the inference demo.

The full WSJ0, WSJ0-2mix, AISHELL-1, and AISHELL-2mix datasets are not included. Dataset partitions, source file lists, data preparation scripts, mixture generation scripts, and full evaluation scripts are also not included.

The original speech data remain subject to the terms of their respective dataset licenses.

## Repository Scope

This repository is intended only for SILC-TSE inference and qualitative listening evaluation. It contains the ONNX inference model, a minimal CPU inference script, and three curated demo examples. Model architecture details, training settings, dataset construction details, and quantitative experimental results are described in the accompanying paper.
