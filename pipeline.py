"""
ASR Benchmark Pipeline
Runs Deepgram Nova-2, Sarvam Saaras v3, and Whisper medium
across all 22 recordings (20 Hindi/Hinglish + 2 English).
IndicConformer runs separately on Colab — results merged via combine_results().

Language detection: files with 'english' in filename use English language codes.

# NOTE: IndicConformer (600M) runs separately on Google Colab T4.
# Pre-computed results: results/indicconformer_results.csv
# To reproduce: run indicconformer_colab.ipynb on Colab with T4 GPU.
# combine_results() merges all 4 model outputs including IndicConformer.
"""

import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from metrics import compute_all_metrics

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
RECORDINGS_DIR = Path("recordings")
TRANSCRIPTS_DIR = Path("transcripts")
RESULTS_DIR = Path("results")
METADATA_PATH = RECORDINGS_DIR / "metadata.csv"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load metadata ─────────────────────────────────────────────────────────────
metadata = pd.read_csv(METADATA_PATH)
metadata["stem"] = metadata["filename"].str.replace(".wav", "", regex=False)


def load_reference(stem: str) -> str:
    path = TRANSCRIPTS_DIR / f"{stem}.txt"
    return path.read_text(encoding="utf-8").strip()


def get_meta(stem: str) -> dict:
    match = metadata[metadata["stem"].str.lower() == stem.lower()]
    if match.empty:
        raise ValueError(f"No metadata found for: {stem}")
    row = match.iloc[0]
    return {
        "locality": row["locality"],
        "condition": row["condition"],
        "gender": row["gender"],
    }


def is_english(stem: str) -> bool:
    """Returns True if the file is an English recording."""
    return "english" in stem.lower()


# ── Deepgram Nova-2 ───────────────────────────────────────────────────────────
def run_deepgram():
    from deepgram import DeepgramClient

    print("\n=== Deepgram Nova-2 (22 files) ===")
    client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
    results = []

    for wav in tqdm(sorted(RECORDINGS_DIR.glob("*.wav"))):
        stem = wav.stem
        ref = load_reference(stem)
        meta = get_meta(stem)

        # English files use "en", Hindi/Hinglish use "hi-Latn" for Roman script
        lang_code = "en" if is_english(stem) else "hi-Latn"

        try:
            audio_data = wav.read_bytes()
            start = time.time()
            response = client.listen.v1.media.transcribe_file(
                request=audio_data,
                model="nova-2",
                language=lang_code,
            )
            latency = time.time() - start
            hyp = response.results.channels[0].alternatives[0].transcript

        except Exception as e:
            print(f"  ERROR on {wav.name}: {e}")
            hyp = ""
            latency = None

        row = compute_all_metrics(
            reference=ref,
            hypothesis=hyp,
            locality=meta["locality"],
            model="deepgram-nova2",
            filename=wav.name,
            condition=meta["condition"],
            gender=meta["gender"],
            latency=latency,
        )
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "deepgram_results.csv", index=False)
    print(f"  Saved {len(df)} rows → results/deepgram_results.csv")
    return df


# ── Sarvam Saaras v3 ──────────────────────────────────────────────────────────
def run_sarvam():
    from sarvamai import SarvamAI

    print("\n=== Sarvam Saaras v3 (22 files) ===")
    client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
    results = []

    for wav in tqdm(sorted(RECORDINGS_DIR.glob("*.wav"))):
        stem = wav.stem
        ref = load_reference(stem)
        meta = get_meta(stem)

        # English files use "en-IN", Hindi/Hinglish use "hi-IN"
        lang_code = "en-IN" if is_english(stem) else "hi-IN"

        try:
            with open(wav, "rb") as f:
                start = time.time()
                response = client.speech_to_text.transcribe(
                    file=f,
                    model="saaras:v3",
                    mode="translit",
                    language_code=lang_code,
                )
                latency = time.time() - start
            hyp = response.transcript

        except Exception as e:
            print(f"  ERROR on {wav.name}: {e}")
            hyp = ""
            latency = None

        row = compute_all_metrics(
            reference=ref,
            hypothesis=hyp,
            locality=meta["locality"],
            model="sarvam-saaras-v3",
            filename=wav.name,
            condition=meta["condition"],
            gender=meta["gender"],
            latency=latency,
        )
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "sarvam_results.csv", index=False)
    print(f"  Saved {len(df)} rows → results/sarvam_results.csv")
    return df


# ── Whisper medium (faster-whisper) ───────────────────────────────────────────
def run_whisper():
    from faster_whisper import WhisperModel

    print("\n=== Whisper medium (faster-whisper, 22 files) ===")
    print("  Loading model...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("  Running on CPU (int8) — ~10s per file")

    results = []

    for wav in tqdm(sorted(RECORDINGS_DIR.glob("*.wav"))):
        stem = wav.stem
        ref = load_reference(stem)
        meta = get_meta(stem)

        # English files use "en"; Hindi/Hinglish use "hi" (outputs Devanagari)
        lang = "en" if is_english(stem) else "hi"

        try:
            start = time.time()
            segments, info = model.transcribe(
                str(wav),
                language=lang,
                beam_size=5,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            hyp = " ".join(seg.text.strip() for seg in segments).lower()
            latency = time.time() - start

        except Exception as e:
            print(f"  ERROR on {wav.name}: {e}")
            hyp = ""
            latency = None

        row = compute_all_metrics(
            reference=ref,
            hypothesis=hyp,
            locality=meta["locality"],
            model="whisper-medium",
            filename=wav.name,
            condition=meta["condition"],
            gender=meta["gender"],
            latency=latency,
        )
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "whisper_results.csv", index=False)
    print(f"  Saved {len(df)} rows → results/whisper_results.csv")
    return df


# ── Combine all results ────────────────────────────────────────────────────────
def combine_results():
    """
    Merges all *_results.csv files (including indicconformer_results.csv from Colab).
    Deduplicates on (filename, model). Prints overall, condition, and gender summaries.
    Expected: 88 rows total (22 files × 4 models).
    """
    from metrics import summarize_results, summarize_by_condition, summarize_by_gender

    csvs = [f for f in RESULTS_DIR.glob("*_results.csv") if f.name != "combined_results.csv"]
    if not csvs:
        print("No result CSVs found in results/")
        return

    dfs = []
    for f in csvs:
        df = pd.read_csv(f)
        print(f"  {f.name} → {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["filename", "model"])
    combined.to_csv(RESULTS_DIR / "combined_results.csv", index=False)
    print(f"\n  combined_results.csv → {len(combined)} rows (expected 88)")

    print("\n=== Overall Summary ===")
    print(summarize_results(combined).to_string())

    print("\n=== By Condition ===")
    print(summarize_by_condition(combined).to_string())

    print("\n=== By Gender ===")
    print(summarize_by_gender(combined).to_string())

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_deepgram()
    run_sarvam()
    run_whisper()
    combine_results()