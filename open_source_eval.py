"""
open_source_eval.py
Evaluates Deepgram Nova-2 and Sarvam Saaras v3 on open-source Hindi datasets.
Datasets: FLEURS Hindi (20 samples), Mozilla Common Voice Hindi (20 samples)
Note: Kathbath excluded — gated dataset requiring HuggingFace approval.
Metrics: WER, CER only — no entity accuracy (not locality recordings)
Results saved to results/open_source_results.csv
"""

import os
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

import io
import time
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Audio

from metrics import compute_wer, compute_cer

load_dotenv()

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES = 20


# ── Audio helpers ──────────────────────────────────────────────────────────────

def array_to_wav_bytes(array: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes for API submission."""
    buf = io.BytesIO()
    sf.write(buf, array, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def save_temp_wav(array: np.ndarray, sample_rate: int) -> str:
    """Save audio array to a temp WAV file. Returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, array, sample_rate)
    return tmp.name


# ── Deepgram inference ─────────────────────────────────────────────────────────

def transcribe_deepgram(audio_bytes: bytes, client) -> tuple:
    """Returns (transcript, latency_sec)."""
    try:
        start = time.time()
        response = client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-2",
            language="hi-Latn",
        )
        latency = time.time() - start
        hyp = response.results.channels[0].alternatives[0].transcript
        return hyp, round(latency, 3)
    except Exception as e:
        print(f"    Deepgram error: {e}")
        return "", None


# ── Sarvam inference ───────────────────────────────────────────────────────────

def transcribe_sarvam(wav_path: str, client) -> tuple:
    """Returns (transcript, latency_sec)."""
    try:
        with open(wav_path, "rb") as f:
            start = time.time()
            response = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                mode="translit",
                language_code="hi-IN",
            )
            latency = time.time() - start
        return response.transcript, round(latency, 3)
    except Exception as e:
        print(f"    Sarvam error: {e}")
        return "", None


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_fleurs(n: int = N_SAMPLES):
    print(f"\nLoading FLEURS Hindi ({n} samples)...")
    dataset = load_dataset(
        "google/fleurs",
        "hi_in",
        split="test",
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    samples = []
    for item in dataset:
        if len(samples) >= n:
            break
        try:
            raw = item["audio"]["bytes"]
            arr, sr = sf.read(io.BytesIO(raw))
            arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)  # stereo → mono
            if arr.size == 0:
                continue
            samples.append({
                "audio_array": arr,
                "sample_rate": sr,
                "reference": item["transcription"].strip(),
                "dataset": "fleurs-hi",
            })
        except Exception as e:
            print(f"  Skipping sample: {e}")
            continue
    print(f"  Loaded {len(samples)} FLEURS samples")
    return samples


def load_common_voice(n: int = N_SAMPLES):
    """
    Load N samples from Mozilla Common Voice Hindi test split.
    Public dataset — no authentication required.
    Used as substitute for Kathbath (gated).
    """
    print(f"\nLoading Mozilla Common Voice Hindi ({n} samples)...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "hi",
        split="test",
        streaming=True,
    )
    samples = []
    for item in dataset:
        if len(samples) >= n:
            break
        arr = np.array(item["audio"]["array"], dtype=np.float32)
        if arr.size == 0:
            continue
        samples.append({
            "audio_array": arr,
            "sample_rate": item["audio"]["sampling_rate"],
            "reference": item["sentence"].strip(),
            "dataset": "common-voice-hi",
        })
    print(f"  Loaded {len(samples)} Common Voice samples")
    return samples


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_open_source_eval():
    from deepgram import DeepgramClient
    from sarvamai import SarvamAI

    deepgram_client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
    sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    try:
        fleurs_samples = load_fleurs(N_SAMPLES)
    except Exception as e:
        print(f"FLEURS load failed: {e}")
        fleurs_samples = []

    try:
        cv_samples = load_common_voice(N_SAMPLES)
    except Exception as e:
        print(f"Common Voice load failed: {e}")
        cv_samples = []

    all_samples = fleurs_samples + cv_samples
    if not all_samples:
        print("No samples loaded. Check internet connection.")
        return

    print(f"\nTotal samples to evaluate: {len(all_samples)}")
    results = []

    for i, sample in enumerate(tqdm(all_samples)):
        arr = sample["audio_array"]
        sr = sample["sample_rate"]
        ref = sample["reference"]
        dataset_name = sample["dataset"]

        audio_bytes = array_to_wav_bytes(arr, sr)
        tmp_path = save_temp_wav(arr, sr)

        print(f"\n[{i+1}/{len(all_samples)}] {dataset_name} | ref: {ref[:60]}")

        # ── Deepgram ──
        dg_hyp, dg_latency = transcribe_deepgram(audio_bytes, deepgram_client)
        dg_wer = compute_wer(ref, dg_hyp)
        dg_cer = compute_cer(ref, dg_hyp)
        print(f"  Deepgram : {dg_hyp[:60]} | WER={dg_wer:.3f} CER={dg_cer:.3f}")

        results.append({
            "dataset": dataset_name,
            "sample_index": i,
            "model": "deepgram-nova2",
            "reference": ref,
            "hypothesis": dg_hyp,
            "wer": dg_wer,
            "cer": dg_cer,
            "latency_sec": dg_latency,
        })

        # ── Sarvam ──
        sv_hyp, sv_latency = transcribe_sarvam(tmp_path, sarvam_client)
        sv_wer = compute_wer(ref, sv_hyp)
        sv_cer = compute_cer(ref, sv_hyp)
        print(f"  Sarvam   : {sv_hyp[:60]} | WER={sv_wer:.3f} CER={sv_cer:.3f}")

        results.append({
            "dataset": dataset_name,
            "sample_index": i,
            "model": "sarvam-saaras-v3",
            "reference": ref,
            "hypothesis": sv_hyp,
            "wer": sv_wer,
            "cer": sv_cer,
            "latency_sec": sv_latency,
        })

        os.unlink(tmp_path)

    # ── Save + summarize ──
    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "open_source_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")

    print("\n=== Open-Source Eval Summary ===")
    summary = df.groupby(["dataset", "model"]).agg(
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        mean_latency=("latency_sec", "mean"),
        count=("sample_index", "count"),
    ).round(4)
    print(summary.to_string())

    return df


if __name__ == "__main__":
    run_open_source_eval()