import re
import pandas as pd
from jiwer import wer, cer

from rapidfuzz import fuzz


LOCALITY_ALIASES = {
    "koramangala": ["koramangala", "कोरमंगला", "करमगल", "कोरमांगला"],
    "indiranagar": ["indiranagar", "indira nagar", "इंदिरानगर", "इदरनगर"],
    "whitefield": ["whitefield", "white field", "वाइटफील्ड", "वइटफलड"],
    "electronic city": ["electronic city", "electroniccity", "इलेक्ट्रॉनिक सिटी"],
    "marathahalli": ["marathahalli", "मराठाहल्ली", "मरथहलल", "मरतल"],
    "jayanagar": ["jayanagar", "jaya nagar", "जयनगर", "जनकर"],
    "rajajinagar": ["rajajinagar", "rajaji nagar", "राजाजीनगर", "रजधनगर", "रयजतनकर"],
    "hebbal": ["hebbal", "हेब्बल", "हबल"],
    "yelahanka": ["yelahanka", "येलहंका", "यलहग", "यलहंक"],
    "banashankari": ["banashankari", "bana shankari", "बनशंकरी", "बनशगर"],
    "hsr layout": ["hsr layout", "hsrlayout", "h s r layout", "एचएसआर लेआउट", "एचसर ल वट"],
    "btm layout": ["btm layout", "btmlayout", "b t m layout", "बीटीएम लेआउट", "बपएम लआउट"],
    "majestic": ["majestic", "मैजेस्टिक", "मजसचक"],
    "silk board": ["silk board", "silkboard", "सिल्क बोर्ड", "सलक बरड"],
    "bellandur": ["bellandur", "बेल्लंदूर", "बलनदर", "बलदर"],
    "sarjapur": ["sarjapur", "sarjapur road", "सरजापुर", "सरचरप"],
    "bommanahalli": ["bommanahalli", "bommana halli", "बोम्मनहल्ली", "बममनहल", "बममन हलल"],
    "kr puram": ["kr puram", "krpuram", "k r puram", "krishnarajapuram", "केआर पुरम", "क आरपरम"],
    "peenya": ["peenya", "pinia", "पीण्या", "पनय", "पनय"],
    "yeshwanthpur": ["yeshwanthpur", "yashwanthpur", "यशवंतपुर", "इसतनदबरपषय"],
}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate between 0 and 1."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0
    return round(wer(ref, hyp), 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate between 0 and 1."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0
    return round(cer(ref, hyp), 4)



def check_entity_accuracy(locality: str, hypothesis: str, threshold: int = 85) -> bool:
    """
    Check if locality name was correctly captured.
    Uses fuzzy matching to handle ASR variations like 
    'koramaangala' vs 'koramangala'.
    """
    hyp = normalize(hypothesis)
    locality_key = locality.lower().strip()
    
    aliases = LOCALITY_ALIASES.get(locality_key, [locality_key])
    
    for alias in aliases:
        # exact match first
        if alias in hyp:
            return True
        # fuzzy match against each word and bigram in hypothesis
        words = hyp.split()
        alias_words = alias.split()
        alias_len = len(alias_words)
        
        # slide a window of alias length across hypothesis words
        for i in range(len(words) - alias_len + 1):
            window = " ".join(words[i:i + alias_len])
            score = fuzz.ratio(alias, window)
            if score >= threshold:
                return True
    
    return False


def compute_all_metrics(
    reference: str,
    hypothesis: str,
    locality: str,
    model: str,
    filename: str,
    condition: str,
    gender: str,
    latency: float = None,
) -> dict:
    """Compute all metrics for a single sample and return as dict."""
    return {
        "filename": filename,
        "model": model,
        "locality": locality,
        "condition": condition,
        "gender": gender,
        "reference": normalize(reference),
        "hypothesis": normalize(hypothesis),
        "wer": compute_wer(reference, hypothesis),
        "cer": compute_cer(reference, hypothesis),
        "entity_correct": check_entity_accuracy(locality, hypothesis),
        "latency_sec": round(latency, 3) if latency is not None else None,
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model."""
    summary = df.groupby("model").agg(
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        entity_accuracy=("entity_correct", "mean"),
        mean_latency=("latency_sec", "mean"),
        sample_count=("filename", "count"),
    ).round(4)
    return summary


def summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model and noise condition."""
    return df.groupby(["model", "condition"]).agg(
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        entity_accuracy=("entity_correct", "mean"),
        sample_count=("filename", "count"),
    ).round(4)


def summarize_by_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model and speaker gender."""
    return df.groupby(["model", "gender"]).agg(
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        entity_accuracy=("entity_correct", "mean"),
        sample_count=("filename", "count"),
    ).round(4)


if __name__ == "__main__":
    # Quick sanity test
    ref = "haan bhai main koramangala mein rehta hoon near sony world signal"
    hyp = "han bhai main koramangala mein rehta hun near sony world signal"
    locality = "koramangala"

    result = compute_all_metrics(
        reference=ref,
        hypothesis=hyp,
        locality=locality,
        model="test",
        filename="01_koramangala_quiet.wav",
        condition="quiet",
        gender="male",
        latency=0.5,
    )

    for k, v in result.items():
        print(f"{k}: {v}")