"""
open_source_eval.py
Evaluates Deepgram Nova-2 and Sarvam Saaras v3 on open-source Hindi datasets.

Datasets:
  - FLEURS Hindi   : google/fleurs, hi_in, test split (20 samples)
  - Kathbath Hindi : ai4bharat/Kathbath, hindi, valid split (20 samples)
                     Gated — requires HF_TOKEN in .env
                     Audio in audio_filepath column as FLAC bytes
                     Reference transcript in text column

Note on WER: References are Devanagari, model outputs are Roman script.
WER ~1.0 expected — script mismatch, not a failure.
CER and latency are the valid comparison metrics.

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
from huggingface_hub import hf_hub_download

from metrics import compute_wer, compute_cer

load_dotenv()

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES = 20
HF_TOKEN = os.getenv("HF_TOKEN")


# ── Audio helpers ──────────────────────────────────────────────────────────────

def array_to_wav_bytes(array: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes for Deepgram API."""
    buf = io.BytesIO()
    sf.write(buf, array, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def save_temp_wav(array: np.ndarray, sample_rate: int) -> str:
    """Save audio array to temp WAV file. Returns file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, array, sample_rate)
    return tmp.name


def decode_audio_bytes(raw_bytes: bytes) -> tuple:
    """
    Decode raw audio bytes (FLAC/WAV/MP3) to numpy array.
    Returns (array, sample_rate).
    """
    buf = io.BytesIO(raw_bytes)
    arr, sr = sf.read(buf)
    arr = arr.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)  # stereo to mono
    return arr, sr


# ── Model inference ────────────────────────────────────────────────────────────

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

def load_fleurs(n: int = N_SAMPLES) -> list:
    """
    FLEURS Hindi test split.
    Clean read speech, diverse regional accents. Public — no token needed.
    """
    print(f"\nLoading FLEURS Hindi ({n} samples)...")
    try:
        from datasets import load_dataset, Audio
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
                arr, sr = decode_audio_bytes(raw)
                if arr.size == 0:
                    continue
                samples.append({
                    "audio_array": arr,
                    "sample_rate": sr,
                    "reference": item["transcription"].strip(),
                    "dataset": "fleurs-hi",
                })
            except Exception as e:
                print(f"  Skipping FLEURS sample: {e}")
                continue
        print(f"  Loaded {len(samples)} FLEURS samples")
        return samples
    except Exception as e:
        print(f"  FLEURS load failed: {e}")
        return []


def load_kathbath(n: int = N_SAMPLES) -> list:
    """
    Kathbath Hindi validation split.
    Conversational speech, 1218 contributors, 203 districts across India.
    Gated — requires HF_TOKEN.

    Audio stored as FLAC bytes in audio_filepath column.
    Reference transcript in text column.
    Downloaded directly via parquet — avoids torchcodec dependency.
    """
    print(f"\nLoading Kathbath Hindi ({n} samples from parquet)...")

    if not HF_TOKEN:
        print("  HF_TOKEN not found in .env — skipping Kathbath")
        return []

    try:
        # Download parquet file directly — already cached from field check
        parquet_file = hf_hub_download(
            repo_id="ai4bharat/Kathbath",
            filename="hindi/valid-00000-of-00002.parquet",
            repo_type="dataset",
            token=HF_TOKEN,
        )

        df = pd.read_parquet(parquet_file)
        print(f"  Parquet loaded: {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")

        samples = []
        skipped = 0

        for _, row in df.iterrows():
            if len(samples) >= n:
                break
            try:
                # Audio is in audio_filepath column as bytes dict
                audio_data = row["audio_filepath"]

                if isinstance(audio_data, dict):
                    raw_bytes = audio_data.get("bytes")
                elif isinstance(audio_data, bytes):
                    raw_bytes = audio_data
                else:
                    skipped += 1
                    continue

                if not raw_bytes:
                    skipped += 1
                    continue

                arr, sr = decode_audio_bytes(raw_bytes)
                if arr.size == 0:
                    skipped += 1
                    continue

                ref = str(row.get("text", "")).strip()
                if not ref:
                    skipped += 1
                    continue

                samples.append({
                    "audio_array": arr,
                    "sample_rate": sr,
                    "reference": ref,
                    "dataset": "kathbath-hi",
                    "gender": str(row.get("gender", "")),
                    "duration": float(row.get("duration", 0)),
                })

            except Exception as e:
                print(f"  Skipping row: {e}")
                skipped += 1
                continue

        print(f"  Loaded {len(samples)} Kathbath samples ({skipped} skipped)")
        return samples

    except Exception as e:
        print(f"  Kathbath load failed: {e}")
        return []


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_open_source_eval():
    from deepgram import DeepgramClient
    from sarvamai import SarvamAI

    deepgram_client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
    sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    # Load existing results — skip already evaluated datasets
    existing_path = RESULTS_DIR / "open_source_results.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing_datasets = existing["dataset"].unique().tolist()
        print(f"Existing results: {existing_datasets}")
    else:
        existing = pd.DataFrame()
        existing_datasets = []

    # Load datasets — skip if already done
    all_samples = []

    if "fleurs-hi" not in existing_datasets:
        all_samples += load_fleurs(N_SAMPLES)
    else:
        print("\nFLEURS already evaluated — skipping")

    if "kathbath-hi" not in existing_datasets:
        all_samples += load_kathbath(N_SAMPLES)
    else:
        print("\nKathbath already evaluated — skipping")

    if not all_samples:
        print("\nAll datasets already evaluated.")
        print("Delete results/open_source_results.csv to rerun.")
        # Still print summary of existing results
        if not existing.empty:
            print("\n=== Existing Open-Source Eval Summary ===")
            summary = existing.groupby(["dataset", "model"]).agg(
                mean_wer=("wer", "mean"),
                mean_cer=("cer", "mean"),
                mean_latency=("latency_sec", "mean"),
                count=("sample_index", "count"),
            ).round(4)
            print(summary.to_string())
        return

    print(f"\nRunning eval on {len(all_samples)} new samples...")
    results = []

    for i, sample in enumerate(tqdm(all_samples)):
        arr = sample["audio_array"]
        sr = sample["sample_rate"]
        ref = sample["reference"]
        dataset_name = sample["dataset"]

        # Convert to WAV for APIs
        try:
            audio_bytes = array_to_wav_bytes(arr, sr)
        except Exception as e:
            print(f"  Audio conversion failed: {e}")
            continue

        tmp_path = save_temp_wav(arr, sr)

        print(f"\n[{i+1}/{len(all_samples)}] {dataset_name} | "
              f"ref: {ref[:60]}")

        # ── Deepgram ──
        dg_hyp, dg_latency = transcribe_deepgram(audio_bytes, deepgram_client)
        dg_wer = compute_wer(ref, dg_hyp)
        dg_cer = compute_cer(ref, dg_hyp)
        print(f"  Deepgram : {dg_hyp[:60]} | "
              f"WER={dg_wer:.3f} CER={dg_cer:.3f}")

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
        print(f"  Sarvam   : {sv_hyp[:60]} | "
              f"WER={sv_wer:.3f} CER={sv_cer:.3f}")

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

    # Merge with existing and save
    new_df = pd.DataFrame(results)
    if not existing.empty and not new_df.empty:
        final_df = pd.concat([existing, new_df], ignore_index=True)
    elif not new_df.empty:
        final_df = new_df
    else:
        final_df = existing

    final_df = final_df.drop_duplicates(
        subset=["dataset", "sample_index", "model"]
    )
    final_df.to_csv(existing_path, index=False)
    print(f"\nSaved {len(final_df)} total rows → {existing_path}")

    # Summary
    print("\n=== Open-Source Eval Summary ===")
    summary = final_df.groupby(["dataset", "model"]).agg(
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        mean_latency=("latency_sec", "mean"),
        count=("sample_index", "count"),
    ).round(4)
    print(summary.to_string())

    return final_df


if __name__ == "__main__":
    run_open_source_eval()