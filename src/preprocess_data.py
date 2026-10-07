"""Data preprocessing and dataset normalization pipeline.

Combines raw articles from PolyglotFakeFacts and HEMT-Fake into a unified,
cleaned, and shuffled dataset saved at data/processed/master_dataset.csv.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
from langdetect import detect

# Mapping from common language names/variations to standard ISO 639-1 two-letter codes
LANGUAGE_MAP = {
    "english": "en",
    "russian": "ru",
    "spanish": "es",
    "german": "de",
    "french": "fr",
    "czech": "cs",
    "romanian": "ro",
    "italian": "it",
    "hungarian": "hu",
    "finnish": "fi",
    "bulgarian": "bg",
    "armenian": "hy",
    "slovak": "sk",
    "swedish": "sv",
    "lithuanian": "lt",
    "azerbaijani": "az",
    "georgian": "ka",
    "arabic": "ar",
    "gujarati": "gu",
    "hindi": "hi",
    "marathi": "mr",
    "telugu": "te",
}


def normalize_language_code(raw_lang: Optional[str], text: Optional[str] = None) -> str:
    """Normalize language string to standard ISO 639-1 code.

    Args:
        raw_lang: Raw language name or code from metadata.
        text: Optional fallback text to run language detection on.

    Returns:
        str: 2-letter ISO language code (defaults to 'en' if indeterminate).
    """
    if raw_lang and isinstance(raw_lang, str):
        clean_lang = raw_lang.strip().lower()
        if clean_lang in LANGUAGE_MAP:
            return LANGUAGE_MAP[clean_lang]
        if len(clean_lang) == 2:
            return clean_lang

    # Attempt automatic language detection if text is provided
    if text and isinstance(text, str) and len(text.strip()) > 10:
        try:
            detected = detect(text.strip())
            return detected
        except Exception:
            pass

    return "en"


def load_polyglot_data(raw_dir: Path, max_chars: Optional[int] = 512) -> pd.DataFrame:
    """Load and normalize PolyglotFakeFacts Excel files.

    Args:
        raw_dir: Directory containing raw data files.
        max_chars: Maximum characters to keep per article text.

    Returns:
        pd.DataFrame: DataFrame with columns ['text', 'label', 'language'].
    """
    records: List[dict] = []
    files_to_check = [
        ("Real.xlsx", 0),
        ("Fake.xlsx", 1),
        ("20.xlsx", None),
        ("80.xlsx", None),
    ]

    for filename, default_label in files_to_check:
        file_path = raw_dir / filename
        if not file_path.exists():
            continue

        try:
            df = pd.read_excel(file_path)
            # Normalize column names to lowercase
            col_map = {c: c.strip().lower() for c in df.columns}
            df = df.rename(columns=col_map)

            # Determine text column
            text_col = None
            for candidate in ["news original text", "news headline", "english translated version"]:
                if candidate in df.columns:
                    text_col = candidate
                    break

            if text_col is None:
                continue

            # Determine label column
            label_col = "label" if "label" in df.columns else None
            lang_col = "language" if "language" in df.columns else None

            for _, row in df.iterrows():
                # Extract text
                text_val = row.get(text_col)
                if pd.isna(text_val) or not str(text_val).strip():
                    if "news headline" in row and pd.notna(row["news headline"]):
                        text_val = row["news headline"]
                    else:
                        continue

                text_str = str(text_val).strip()
                if not text_str:
                    continue

                if max_chars:
                    text_str = text_str[:max_chars].strip()

                # Extract label
                if default_label is not None:
                    label = default_label
                elif label_col and pd.notna(row.get(label_col)):
                    raw_lbl = str(row[label_col]).strip().lower()
                    if raw_lbl in ["real", "0"]:
                        label = 0
                    elif raw_lbl in ["fake", "1"]:
                        label = 1
                    else:
                        continue
                else:
                    continue

                # Extract language
                raw_lang = row.get(lang_col) if lang_col else None
                lang_code = normalize_language_code(raw_lang, text_str)

                records.append({
                    "text": text_str,
                    "label": int(label),
                    "language": lang_code,
                })
        except Exception as e:
            print(f"Warning: Failed reading {filename}: {e}")

    return pd.DataFrame(records)


def load_hemt_data(raw_dir: Path, max_chars: Optional[int] = 512) -> pd.DataFrame:
    """Load and normalize HEMT-Fake dataset folders.

    Args:
        raw_dir: Directory containing raw data folders.
        max_chars: Maximum characters to keep per article text.

    Returns:
        pd.DataFrame: DataFrame with columns ['text', 'label', 'language'].
    """
    records: List[dict] = []
    hemt_configs = [
        ("Gujarati_fake_news", 1, "gu"),
        ("Gujarati_real_news", 0, "gu"),
        ("Hindi_fake_news", 1, "hi"),
        ("Hindi_real_news", 0, "hi"),
        ("Marathi_fake_news", 1, "mr"),
        ("Marathi_real_news", 0, "mr"),
        ("Telugu_fake_news", 1, "te"),
        ("Telugu_real_news", 0, "te"),
    ]

    for folder_name, label, lang_code in hemt_configs:
        folder_path = raw_dir / folder_name
        if not folder_path.exists():
            continue

        for txt_file in folder_path.glob("*.txt"):
            try:
                content = txt_file.read_text(encoding="utf-8", errors="replace").strip()
                if not content:
                    continue
                if max_chars:
                    content = content[:max_chars].strip()
                if not content:
                    continue

                records.append({
                    "text": content,
                    "label": label,
                    "language": lang_code,
                })
            except Exception:
                continue

    return pd.DataFrame(records)


def preprocess_and_merge(
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("data/processed/master_dataset.csv"),
    max_chars: Optional[int] = 512,
    random_state: int = 42,
) -> pd.DataFrame:
    """Preprocess, normalize, deduplicate, and shuffle raw datasets into master CSV.

    Args:
        raw_dir: Directory where raw dataset files are stored.
        output_path: Target path for the unified CSV.
        max_chars: Character truncation limit for text articles.
        random_state: Random seed for shuffling.

    Returns:
        pd.DataFrame: The final cleaned and shuffled DataFrame.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    print(f"Loading PolyglotFakeFacts from {raw_dir}...")
    poly_df = load_polyglot_data(raw_dir, max_chars=max_chars)
    print(f"Loaded {len(poly_df)} records from PolyglotFakeFacts.")

    print(f"Loading HEMT-Fake from {raw_dir}...")
    hemt_df = load_hemt_data(raw_dir, max_chars=max_chars)
    print(f"Loaded {len(hemt_df)} records from HEMT-Fake.")

    # Combine datasets
    df = pd.concat([poly_df, hemt_df], ignore_index=True)

    # Clean data
    # Drop rows with NaNs
    df = df.dropna(subset=["text", "label", "language"])
    # Strip whitespace
    df["text"] = df["text"].astype(str).str.strip()
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(int)

    # Remove empty strings
    df = df[df["text"].str.len() > 0]
    df = df[df["language"].str.len() > 0]

    # Drop duplicates on text to remove subset/split overlaps
    initial_len = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"Deduplicated from {initial_len} to {len(df)} records.")

    # Shuffle randomly
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved master dataset to {output_path} ({len(df)} rows).")

    return df


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = preprocess_and_merge()
    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print("\nLabel Distribution:")
    print(df["label"].value_counts())
    print("\nLanguage Distribution:")
    print(df["language"].value_counts())
    print("\nFirst 5 Rows:")
    print(df.head())
