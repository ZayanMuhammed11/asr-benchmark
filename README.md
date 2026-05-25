# ASR Benchmark — Indian Conversational Speech

**Evaluating automatic speech recognition systems for locality name extraction from Hindi/Hinglish phone speech.**

Built for a blue-collar hiring platform where candidates report their location via phone call or WhatsApp voice note. The core task is not transcription accuracy — it is entity extraction: did the model correctly capture the locality name? Wrong locality = wrong job match.

---

## Results Summary

| Model | Entity Accuracy | Mean WER | Mean Latency |
|---|---|---|---|
| Deepgram Nova-2 | 32% | 0.49 | 0.9s |
| **Sarvam Saaras v3** | **77%** | **0.25** | **0.7s** |
| IndicConformer 600M | 55% | 1.01* | 1.5s |
| Whisper Medium | 64% | 0.94* | 12.4s |

*WER not comparable — Devanagari output vs Roman reference

**Recommendation:** Sarvam Saaras v3 → Llama-3 NER → Gazetteer validation (95% entity accuracy)

Full findings in [report.docx](report.docx)

---

## Dataset

- 22 phone-mic recordings — 20 Hindi/Hinglish + 2 English
- Bangalore locality names in natural conversational sentences
- Two speakers: male (16 files), female (6 files), Malayalam-accented Hindi
- Conditions: quiet (8), noisy (9), rushed (3), whispered (2)
- External validation: FLEURS Hindi test split (20 samples)

---

## Setup

```bash
# Create and activate virtual environment
python -m venv vaah
vaah\Scripts\activate        # Windows
source vaah/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your keys
```

---

## API Keys Required

| Key | Get it from |
|---|---|
| DEEPGRAM_API_KEY | deepgram.com — free tier |
| SARVAM_API_KEY | sarvam.ai — free tier |
| GROQ_API_KEY | console.groq.com — free tier |
| HF_TOKEN | huggingface.co/settings/tokens |

---

## Run

```bash
# Full benchmark — Deepgram + Sarvam + Whisper on all 22 files
python pipeline.py

# Generate all 5 charts
python analyse.py

# LLM extraction comparison (ASR output → Llama-3 NER)
python llm_extraction.py

# Open-source dataset evaluation (FLEURS Hindi)
python open_source_eval.py

# Generate report.docx
python generate_report.py

# Verify all numbers match CSVs
python verify_submission.py
```

---

## IndicConformer

Runs on Google Colab T4 GPU — model is 6.17GB, exceeds local 6GB VRAM.
Pre-computed results are in `results/indicconformer_results.csv`.
To reproduce: open `indic_conformer.ipynb` in Google Colab (T4 runtime) and run all cells.

---

## Project Structure

| File / Folder | Purpose |
|---|---|
| `recordings/` | 22 WAV files + metadata.csv |
| `transcripts/` | 22 reference TXT files (one per WAV) |
| `results/combined_results.csv` | 88 rows — 22 files × 4 models |
| `results/charts/` | 5 PNG charts |
| `pipeline.py` | Inference — Deepgram + Sarvam + Whisper on all 22 files |
| `metrics.py` | WER, CER, entity accuracy, fuzzy matching |
| `analyse.py` | Chart generation from combined_results.csv |
| `llm_extraction.py` | ASR → Llama-3 NER two-stage pipeline |
| `open_source_eval.py` | FLEURS Hindi external evaluation |
| `process_indicconformer.py` | Merges Colab IndicConformer output with metadata |
| `generate_report.py` | Generates report.docx |
| `verify_submission.py` | Cross-checks all numbers against CSVs |
| `indic_conformer.ipynb` | IndicConformer Colab notebook (T4 GPU) |
| `project_log.md` | Full decision log — every choice and tradeoff |
| `requirements.txt` | Python dependencies |
| `.env.example` | API key template |
| `report.docx` | Final benchmark report |

---

## Key Findings

- **Sarvam is the only model handling rushed audio** — 100% vs 0% for Deepgram and IndicConformer
- **Deepgram produced empty outputs** on whispered and noisy audio — no other model did
- **Whisper scores 100% on whispered audio** but is not deployable at 12.4s latency
- **LLM post-processing** improves Deepgram 32%→73%, Sarvam 77%→95%
- **Sarvam gender gap**: 87% male vs 57% female — production risk, needs monitoring
- **IndicConformer ties Sarvam on quiet audio** (86%) but collapses on noise and rushed speech

---

## Models

| Model | Type | HuggingFace / API |
|---|---|---|
| Deepgram Nova-2 | API | deepgram.com |
| Sarvam Saaras v3 | API | sarvam.ai |
| Whisper Medium | Local | openai/whisper-medium |
| IndicConformer 600M | Local (Colab) | ai4bharat/indicconformer — gated, request access |

---

## Technical Notes

- `faster-whisper` used instead of `openai-whisper` — Python 3.12 compatibility
- Deepgram `language="hi-Latn"` — `"hi"` returns Devanagari, breaks WER comparison
- Sarvam `mode="translit"` — `"transcribe"` returns Devanagari, WER=1.0 for all files
- Whisper outputs Devanagari for Hindi — Devanagari aliases added to entity matcher
- Fuzzy match threshold 85 not 80 — "electronic city" matched "electricity" at 84.6
- - Whisper ran on CPU (int8 quantization) — GTX 1660 missing CUDA cublas DLL. 
  GPU inference validated on tiny model only. Latency of 12.4s is a CPU baseline; 
  GPU would be ~1-2s but Devanagari output issue and Sarvam's 0.7s latency 
  still make Whisper non-competitive for this use case.
