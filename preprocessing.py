import re
import string
import emoji
import pandas as pd

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw message text and extracts metadata features."""
    df = df.copy()
    df["message_text"] = df["message_text"].fillna("").apply(clean_text)
    
    # Feature engineering
    df["caps_ratio"] = df["message_text"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / len(t) if len(t) > 0 else 0.0
    )
    df["punctuation_count"] = df["message_text"].apply(
        lambda t: sum(1 for c in t if c in string.punctuation)
    )
    df["message_length"] = df["message_text"].str.len()
    
    return df

def clean_text(text: str) -> str:
    text = emoji.demojize(str(text))
    return re.sub(r"[^\w\s" + re.escape(string.punctuation) + r"]", "", text)