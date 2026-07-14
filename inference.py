import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SR = 8000
TARGET_SAMPLES = 32000
EXPECTED_INPUTS = {
    "mixture": [1, 1, TARGET_SAMPLES],
    "reference": [1, 1, TARGET_SAMPLES],
}
EXPECTED_OUTPUT = {
    "estimate": [1, 1, TARGET_SAMPLES],
}


class InferenceError(RuntimeError):
    pass


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise InferenceError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise InferenceError(f"{label} path is not a file: {path}")
    return path


def read_audio(path, label):
    try:
        audio, sr = sf.read(str(path), always_2d=False)
    except Exception as exc:
        raise InferenceError(f"Could not read {label} audio: {path}. {exc}") from exc
    audio = np.asarray(audio)
    if audio.size == 0:
        raise InferenceError(f"{label} audio is empty: {path}")
    return audio, int(sr)


def to_mono(audio):
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    raise InferenceError(f"Unsupported audio shape: {audio.shape}")


def resample_audio(audio, sr):
    if sr == TARGET_SR:
        return audio.astype(np.float32)
    gcd = math.gcd(int(sr), TARGET_SR)
    return resample_poly(audio, TARGET_SR // gcd, int(sr) // gcd).astype(np.float32)


def fix_length(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[0] < TARGET_SAMPLES:
        return np.pad(audio, (0, TARGET_SAMPLES - audio.shape[0])).astype(np.float32)
    return audio[:TARGET_SAMPLES].astype(np.float32)


def prepare_mixture(audio, sr):
    audio = fix_length(resample_audio(to_mono(audio), sr))
    max_abs = float(np.max(np.abs(audio)))
    if max_abs >= 1.0:
        audio = audio / max_abs
    return audio.astype(np.float32).reshape(1, 1, TARGET_SAMPLES)


def prepare_reference(audio, sr):
    audio = fix_length(resample_audio(to_mono(audio), sr))
    std = float(np.std(audio))
    if std < 1e-12:
        raise InferenceError("Reference standard deviation is too small after preprocessing.")
    audio = (audio - np.mean(audio)) / (std + 1e-8)
    return audio.astype(np.float32).reshape(1, 1, TARGET_SAMPLES)


def normalize_shape(shape):
    return [int(x) if isinstance(x, (int, np.integer)) else x for x in shape]


def check_session(session):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_map = {item.name: item for item in inputs}
    output_map = {item.name: item for item in outputs}

    for name, shape in EXPECTED_INPUTS.items():
        if name not in input_map:
            raise InferenceError(f"ONNX input name mismatch. Missing input: {name}")
        item = input_map[name]
        if item.type != "tensor(float)":
            raise InferenceError(f"ONNX input dtype mismatch for {name}: expected tensor(float), got {item.type}")
        if normalize_shape(item.shape) != shape:
            raise InferenceError(f"ONNX input shape mismatch for {name}: expected {shape}, got {item.shape}")

    for name, shape in EXPECTED_OUTPUT.items():
        if name not in output_map:
            raise InferenceError(f"ONNX output name mismatch. Missing output: {name}")
        item = output_map[name]
        if item.type != "tensor(float)":
            raise InferenceError(f"ONNX output dtype mismatch for {name}: expected tensor(float), got {item.type}")
        if normalize_shape(item.shape) != shape:
            raise InferenceError(f"ONNX output shape mismatch for {name}: expected {shape}, got {item.shape}")


def run(args):
    model_path = require_file(args.model, "Model")
    mixture_path = require_file(args.mixture, "Mixture")
    reference_path = require_file(args.reference, "Reference")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mixture_audio, mixture_sr = read_audio(mixture_path, "mixture")
    reference_audio, reference_sr = read_audio(reference_path, "reference")
    mixture = prepare_mixture(mixture_audio, mixture_sr)
    reference = prepare_reference(reference_audio, reference_sr)

    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        check_session(session)
        start = time.perf_counter()
        estimate = session.run(["estimate"], {"mixture": mixture, "reference": reference})[0]
        elapsed = time.perf_counter() - start
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(f"ONNX Runtime inference failed: {exc}") from exc

    if estimate.shape != (1, 1, TARGET_SAMPLES):
        raise InferenceError(f"Unexpected output shape: {estimate.shape}")
    if np.isnan(estimate).any() or np.isinf(estimate).any():
        raise InferenceError("Model output contains NaN or Inf.")

    waveform = estimate.reshape(-1).astype(np.float32)
    sf.write(str(output_path), waveform, TARGET_SR, subtype="FLOAT")

    duration = TARGET_SAMPLES / TARGET_SR
    rtf = elapsed / duration
    print(f"model path: {model_path}")
    print(f"mixture path: {mixture_path}")
    print(f"reference path: {reference_path}")
    print(f"original mixture sample rate: {mixture_sr}")
    print(f"original reference sample rate: {reference_sr}")
    print(f"processed sample rate: {TARGET_SR}")
    print(f"processed sample count: {TARGET_SAMPLES}")
    print("inference provider: CPUExecutionProvider")
    print(f"inference time: {elapsed:.6f} s")
    print(f"audio duration: {duration:.3f} s")
    print(f"RTF: {rtf:.6f}")
    print(f"output path: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SCI-LFC ONNX inference.")
    parser.add_argument("--mixture", required=True, help="Path to the mixture WAV file.")
    parser.add_argument("--reference", required=True, help="Path to the target-speaker reference WAV file.")
    parser.add_argument("--model", required=True, help="Path to the SCI-LFC ONNX model.")
    parser.add_argument("--output", required=True, help="Path for the output WAV file.")
    args = parser.parse_args()
    try:
        run(args)
    except InferenceError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()

