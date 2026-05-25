"""
analyse.py
Loads combined_results.csv and llm_extraction_comparison.csv.
Generates 5 charts for the report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path("results")
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "combined_results.csv")

MODEL_ORDER = [
    "deepgram-nova2",
    "sarvam-saaras-v3",
    "indicconformer-600m",
    "whisper-medium",
]
MODEL_LABELS = {
    "deepgram-nova2":       "Deepgram\nNova-2",
    "sarvam-saaras-v3":     "Sarvam\nSaaras v3",
    "indicconformer-600m":  "IndicConformer\n600M",
    "whisper-medium":       "Whisper\nMedium",
}
COLORS = ["#E07B54", "#4C9BE8", "#6DBE6D", "#B07FD4"]

# ── Chart 1: Entity Accuracy by Model ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
summary = df.groupby("model")["entity_correct"].mean().reindex(MODEL_ORDER)
bars = ax.bar(
    [MODEL_LABELS[m] for m in MODEL_ORDER],
    summary.values * 100,
    color=COLORS,
    edgecolor="white",
    linewidth=0.8,
)
for bar, val in zip(bars, summary.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val*100:.0f}%",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )
ax.set_ylabel("Entity Accuracy (%)", fontsize=12)
ax.set_title("Locality Name Capture Rate by Model", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105)
ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "entity_accuracy.png", dpi=150)
plt.close()
print("Saved entity_accuracy.png")

# ── Chart 2: Entity Accuracy by Condition ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

# Filter to only Hindi/Hinglish files for condition breakdown
# English files (files 21,22) don't have rushed/whispered — skews the chart
hindi_df = df[~df["filename"].str.contains("english", case=False, na=False)]

cond_df = hindi_df.groupby(["model", "condition"])["entity_correct"].mean().reset_index()
cond_pivot = cond_df.pivot(index="condition", columns="model", values="entity_correct")
cond_pivot = cond_pivot.reindex(columns=MODEL_ORDER)
cond_pivot = cond_pivot.reindex(["quiet", "noisy", "rushed", "whispered"])
cond_pivot.plot(
    kind="bar",
    ax=ax,
    color=COLORS,
    edgecolor="white",
    linewidth=0.5,
    width=0.7,
)
ax.set_ylabel("Entity Accuracy", fontsize=12)
ax.set_title("Entity Accuracy by Noise Condition (Hindi/Hinglish only)", fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.set_xticklabels(["Quiet", "Noisy", "Rushed", "Whispered"], rotation=0)
ax.legend(
    [MODEL_LABELS[m] for m in MODEL_ORDER],
    title="Model", fontsize=9, title_fontsize=9
)
ax.set_ylim(0, 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "noise_breakdown.png", dpi=150)
plt.close()
print("Saved noise_breakdown.png")

# ── Chart 3: Latency by Model ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
latency = df.groupby("model")["latency_sec"].mean().reindex(MODEL_ORDER)
bars = ax.bar(
    [MODEL_LABELS[m] for m in MODEL_ORDER],
    latency.values,
    color=COLORS,
    edgecolor="white",
    linewidth=0.8,
)
for bar, val in zip(bars, latency.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{val:.1f}s",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )
ax.set_ylabel("Mean Latency (seconds)", fontsize=12)
ax.set_title("Mean Inference Latency per File", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "latency_comparison.png", dpi=150)
plt.close()
print("Saved latency_comparison.png")

# ── Chart 4: Gender Gap ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
gender_df = df.groupby(["model", "gender"])["entity_correct"].mean().reset_index()
gender_pivot = gender_df.pivot(index="model", columns="gender", values="entity_correct")
gender_pivot = gender_pivot.reindex(MODEL_ORDER)
x = range(len(MODEL_ORDER))
width = 0.35
bars1 = ax.bar(
    [i - width/2 for i in x],
    gender_pivot["male"].values * 100,
    width, label="Male", color="#4C9BE8", edgecolor="white"
)
bars2 = ax.bar(
    [i + width/2 for i in x],
    gender_pivot["female"].values * 100,
    width, label="Female", color="#E07B54", edgecolor="white"
)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Entity Accuracy (%)", fontsize=12)
ax.set_title("Entity Accuracy by Speaker Gender", fontsize=13, fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "gender_breakdown.png", dpi=150)
plt.close()
print("Saved gender_breakdown.png")

# ── Chart 5: LLM vs Fuzzy Extraction ──────────────────────────────────────────
llm_path = RESULTS_DIR / "llm_extraction_comparison.csv"
if llm_path.exists():
    llm_df = pd.read_csv(llm_path)

    # Warn if row count looks wrong
    print(f"\nLLM CSV: {len(llm_df)} rows across models: {llm_df['model'].value_counts().to_dict()}")

    fig, ax = plt.subplots(figsize=(8, 5))

    llm_summary = llm_df.groupby("model").agg(
        fuzzy_acc=("fuzzy_correct", "mean"),
        llm_acc=("llm_correct", "mean"),
    ).reindex(["deepgram-nova2", "sarvam-saaras-v3"])

    api_labels = ["Deepgram\nNova-2", "Sarvam\nSaaras v3"]
    x = range(len(llm_summary))
    width = 0.35

    bars1 = ax.bar(
        [i - width/2 for i in x],
        llm_summary["fuzzy_acc"].values * 100,
        width, label="Fuzzy Match", color="#E07B54", edgecolor="white"
    )
    bars2 = ax.bar(
        [i + width/2 for i in x],
        llm_summary["llm_acc"].values * 100,
        width, label="LLM Extraction (Llama-3)", color="#4C9BE8", edgecolor="white"
    )
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(api_labels)
    ax.set_ylabel("Entity Accuracy (%)", fontsize=12)
    ax.set_title("Fuzzy Matching vs LLM Extraction (Deepgram + Sarvam)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "llm_vs_fuzzy.png", dpi=150)
    plt.close()
    print("Saved llm_vs_fuzzy.png")
else:
    print("llm_extraction_comparison.csv not found — skipping Chart 5")

print("\nAll charts saved to results/charts/")
