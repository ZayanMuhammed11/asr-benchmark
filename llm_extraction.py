"""
llm_extraction.py
Two-stage pipeline: ASR transcript -> LLM locality extraction
Compares fuzzy matching vs LLM-based extraction.
Uses Groq (Llama-3) for fast, free inference.
"""

import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from metrics import check_entity_accuracy

load_dotenv()

RESULTS_DIR = Path("results")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_locality_llm(transcript: str) -> str:
    """
    Stage 2: Pass ASR transcript to Llama-3 via Groq.
    Extract locality name from transcribed text.
    """
    if not transcript or str(transcript).strip() in ["", "nan"]:
        return "NONE"

    prompt = f"""You are an assistant that extracts locality/area/neighbourhood names from Indian speech transcripts.

You are an information extraction system.

Extract the locality, area, or neighborhood name from the transcript.

The transcript may contain:
- Hindi
- Hinglish
- English
- phonetic distortions
- ASR transcription errors

Return ONLY the corrected locality name in Roman script.

Rules:
- Do not explain
- Do not give multiple options
- Do not output sentences
- Output only one locality name
- If unsure, return NONE

Examples:
Text: "main koramangala mein rehta hoon" -> Koramangala
Text: "alhaga mein ek naya flat liya hai" -> Yelahanka
Text: "hanbhai main koramaangala mein rahta hoon" -> Koramangala
Text: "yaar silver par yah bahut hi jaan hai" -> Silk Board
Text: "bpm leut mein hoon yaar" -> BTM Layout
Text: "bailandoor lake ke paas ek pg mein hoon" -> Bellandur
Text: "pine industrial area mein kaam karta hoon" -> Peenya

Text: {transcript}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Groq error: {e}")
        return "ERROR"


def run_comparison():
    df = pd.read_csv(RESULTS_DIR / "combined_results.csv")

    # Use only Roman script models
    api_models = df[df["model"].isin([
        "deepgram-nova2",
        "sarvam-saaras-v3"
    ])].copy()

    print(f"Running LLM extraction on {len(api_models)} samples")
    print("Stage 1: ASR transcript already done")
    print("Stage 2: Passing transcripts to Llama-3 via Groq\n")

    results = []

    for _, row in api_models.iterrows():
        hyp = str(row["hypothesis"]) if pd.notna(row["hypothesis"]) else ""
        locality = row["locality"]
        fuzzy_result = bool(row["entity_correct"])

        # Stage 2 — LLM extraction
        llm_extracted = extract_locality_llm(hyp)
        llm_correct = check_entity_accuracy(locality, llm_extracted)

        print(f"[{row['model']:<20}] {row['filename']}")
        print(f"  Locality : {locality}")
        print(f"  ASR out  : {hyp[:60]}")
        print(f"  Fuzzy    : {'✓' if fuzzy_result else '✗'}")
        print(f"  LLM got  : {llm_extracted}")
        print(f"  LLM      : {'✓' if llm_correct else '✗'}")
        print()

        results.append({
            "filename": row["filename"],
            "model": row["model"],
            "locality": locality,
            "condition": row["condition"],
            "gender": row["gender"],
            "asr_transcript": hyp,
            "fuzzy_correct": fuzzy_result,
            "llm_extracted": llm_extracted,
            "llm_correct": llm_correct,
        })

        # Groq free tier: 30 requests/min — stay safe
        time.sleep(2)

    df_results = pd.DataFrame(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("=== Two-Stage Pipeline Comparison ===")
    print("="*50)
    print(f"Total samples     : {len(df_results)}")
    print(f"Fuzzy accuracy    : {df_results['fuzzy_correct'].mean():.2%}")
    print(f"LLM accuracy      : {df_results['llm_correct'].mean():.2%}")

    rescued = df_results[
        (~df_results["fuzzy_correct"]) & (df_results["llm_correct"])
    ]
    print(f"\nLLM rescued {len(rescued)} fuzzy failures:")
    for _, r in rescued.iterrows():
        print(f"  {r['locality']:20} | ASR: {r['asr_transcript'][:40]:40} | LLM: {r['llm_extracted']}")

    regressed = df_results[
        (df_results["fuzzy_correct"]) & (~df_results["llm_correct"])
    ]
    if len(regressed):
        print(f"\nLLM missed {len(regressed)} fuzzy successes:")
        for _, r in regressed.iterrows():
            print(f"  {r['locality']:20} | LLM got: {r['llm_extracted']}")

    print("\n=== By Model ===")
    summary = df_results.groupby("model").agg(
        fuzzy_acc=("fuzzy_correct", "mean"),
        llm_acc=("llm_correct", "mean"),
        count=("filename", "count")
    ).round(3)
    print(summary.to_string())

    df_results.to_csv(RESULTS_DIR / "llm_extraction_comparison.csv", index=False)
    print("\nSaved → results/llm_extraction_comparison.csv")
    return df_results


if __name__ == "__main__":
    run_comparison()