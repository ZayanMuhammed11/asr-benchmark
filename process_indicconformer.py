
import pandas as pd
from pathlib import Path
from metrics import compute_all_metrics

RECORDINGS_DIR = Path("recordings")
TRANSCRIPTS_DIR = Path("transcripts")
RESULTS_DIR = Path("results")

metadata = pd.read_csv(RECORDINGS_DIR / "metadata.csv")
metadata["stem"] = metadata["filename"].str.replace(".wav", "", regex=False)

raw = pd.read_csv(RESULTS_DIR / "indicconformer_results.csv")
results = []

for _, row in raw.iterrows():
    stem = row["filename"].replace(".wav", "")
    ref_path = TRANSCRIPTS_DIR / f"{stem}.txt"
    ref = ref_path.read_text(encoding="utf-8").strip()
    
    meta_row = metadata[metadata["stem"].str.lower() == stem.lower()].iloc[0]
    
    result = compute_all_metrics(
        reference=ref,
        hypothesis=str(row["hypothesis"]),
        locality=meta_row["locality"],
        model="indicconformer-600m",
        filename=row["filename"],
        condition=meta_row["condition"],
        gender=meta_row["gender"],
        latency=row["latency_sec"],
    )
    results.append(result)

df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / "indicconformer_results.csv", index=False)
print(f"Saved {len(df)} rows")
print(f"Entity accuracy: {df['entity_correct'].mean():.2%}")
print(f"Mean WER: {df['wer'].mean():.4f}")
print(f"Mean latency: {df['latency_sec'].mean():.3f}s")
