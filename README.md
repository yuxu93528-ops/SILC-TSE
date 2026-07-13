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

Target speaker extraction does not have a single universally adopted ready-made public dataset. In the accompanying study, WSJ0-2mix and AISHELL-2mix were constructed from public speech corpora for model evaluation. Each TSE sample consists of a mixture speech signal, a target speech signal, and a reference speech signal. The reference and target speech signals belong to the same speaker but are different utterances.

### WSJ0-2mix

WSJ0-2mix is one of the commonly used datasets for target speaker extraction. The speech signals are derived from the WSJ0 corpus, which contains English speech recordings from Wall Street Journal text. The original recordings are single-channel audio with a sampling rate of 16 kHz.

For training and validation, utterances are taken from the `si_tr_s` subset, which contains 101 speakers, including 50 male speakers and 51 female speakers, with 12,776 utterances in total. For testing, utterances are taken from the `si_dt_05` and `si_et_05` subsets, which contain 18 speakers unseen during training, including 10 male speakers and 8 female speakers, with 1,857 utterances in total.

Each WSJ0-2mix mixture is generated from two different speakers. One utterance is randomly selected for each speaker, downsampled to 8 kHz, and normalized to unit power. The amplitude of one utterance is then adjusted so that the relative energy between the two utterances falls within -5 dB to 5 dB. Because the two utterances may have different durations, the mixture length is determined by the shorter utterance. The two processed utterances are then summed to form the mixture.

The constructed WSJ0-2mix set contains 20,000 training mixtures, 5,000 validation mixtures, and 3,000 test mixtures. To form TSE samples, each mixture is used twice by alternately treating each speaker as the target speaker. The reference speech for a target speaker is selected from another utterance of the same speaker.

### AISHELL-2mix

AISHELL-2mix is constructed from the AISHELL Mandarin speech corpus. AISHELL contains recordings from 400 speakers from different accent regions in China. The original corpus is divided into training, validation, and test subsets containing 340, 40, and 20 speakers, respectively.

Before mixture generation, AISHELL utterances shorter than 2 seconds are discarded. The mixture construction procedure follows the WSJ0-2mix setup, except that the relative energy range is expanded to -6 dB to 6 dB.

The constructed AISHELL-2mix set contains 100,000 training mixtures, 5,000 validation mixtures, and 8,000 test mixtures.

## Demo Data

The audio examples included in this repository are three short, curated qualitative examples derived from the WSJ0-2mix test set. They are provided only for qualitative listening with the inference demo.

The full WSJ0, WSJ0-2mix, AISHELL, and AISHELL-2mix datasets are not included. Dataset partitions, source file lists, data preparation scripts, mixture generation scripts, and full evaluation scripts are also not included.

The original speech data remain subject to the terms of their respective dataset licenses.

## Repository Scope

This repository is intended only for SILC-TSE inference and qualitative listening evaluation. It contains the ONNX inference model, a minimal CPU inference script, and three curated demo examples. Model architecture details, training settings, dataset construction details, and quantitative experimental results are described in the accompanying paper.
