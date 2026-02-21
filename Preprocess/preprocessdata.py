import pandas as pd
import numpy as np

import argparse
import csv
from pathlib import Path

from pandas.errors import ParserError


DEFAULT_INPUT_NAME = "Vegetables_fruit_prices_with_climate_130000_2020_to_2025.csv"


def _resolve_input_csv_path(cli_value: str | None) -> Path:
    if cli_value:
        candidate = Path(cli_value).expanduser()
        if candidate.is_file():
            return candidate
        raise SystemExit(f"ERROR: Input file not found: {candidate}")

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / DEFAULT_INPUT_NAME,
        script_dir / DEFAULT_INPUT_NAME,
        script_dir.parent / DEFAULT_INPUT_NAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    def _csvs_in_dir(directory: Path) -> list[str]:
        try:
            return sorted(p.name for p in directory.glob("*.csv"))
        except OSError:
            return []

    searched = "\n".join(f"- {p}" for p in candidates)
    nearby = {
        "cwd": _csvs_in_dir(Path.cwd()),
        "script_dir": _csvs_in_dir(script_dir),
        "parent": _csvs_in_dir(script_dir.parent),
    }
    raise SystemExit(
        "ERROR: Could not find the input CSV.\n"
        f"Searched:\n{searched}\n\n"
        "Fix: pass the full path, e.g.\n"
        "  python preprocessdata.py --input C:\\path\\to\\file.csv\n\n"
        f"CSV files found nearby: {nearby}"
    )


def _read_csv_with_encoding_fallback(path: Path, encoding: str | None, sep_override: str | None) -> pd.DataFrame:
    sample_bytes = path.read_bytes()[:64_000]

    def _detect_encoding() -> str:
        if sample_bytes.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if sample_bytes.startswith(b"\xff\xfe") or sample_bytes.startswith(b"\xfe\xff"):
            return "utf-16"
        # Heuristic: lots of NUL bytes usually indicates UTF-16.
        if sample_bytes.count(b"\x00") > 20:
            return "utf-16"
        try:
            sample_bytes.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "cp1252"

    def _sniff_sep(text_sample: str) -> str:
        try:
            dialect = csv.Sniffer().sniff(text_sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except Exception:
            return ","

    if encoding:
        encodings_to_try = [encoding]
    else:
        preferred = _detect_encoding()
        # Common encodings for CSVs exported from Excel/Windows tools.
        encodings_to_try = [preferred, "utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]
        # de-dup while preserving order
        seen: set[str] = set()
        encodings_to_try = [e for e in encodings_to_try if not (e in seen or seen.add(e))]

    errors: list[str] = []
    for candidate in encodings_to_try:
        try:
            text_sample = sample_bytes.decode(candidate, errors="replace")
            sep = sep_override or _sniff_sep(text_sample)

            try:
                return pd.read_csv(path, encoding=candidate, sep=sep, low_memory=False)
            except (ParserError, MemoryError) as exc:
                # Retry with the more tolerant Python engine.
                errors.append(f"- {candidate} (sep='{sep}') C-engine parse failed: {exc}")
                try:
                    return pd.read_csv(
                        path,
                        encoding=candidate,
                        sep=sep,
                        engine="python",
                        on_bad_lines="skip",
                    )
                except (ParserError, MemoryError) as exc2:
                    errors.append(f"- {candidate} (sep='{sep}') Python-engine parse failed: {exc2}")
                    continue
        except UnicodeDecodeError as exc:
            errors.append(f"- {candidate}: {exc}")
        except (ParserError, MemoryError) as exc:
            errors.append(f"- {candidate}: {exc}")
        except Exception as exc:
            errors.append(f"- {candidate}: {type(exc).__name__}: {exc}")

    joined = "\n".join(errors)
    raise SystemExit(
        f"ERROR: Could not read CSV file: {path}\nTried encodings and parsers:\n{joined}\n\n"
        "Fix: rerun with explicit options, e.g.\n"
        "  python preprocessdata.py --input <file.csv> --encoding cp1252\n"
        "  python preprocessdata.py --input <file.csv> --encoding utf-16"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dataset and export cleaned carrot-only features.")
    parser.add_argument("--input", "-i", help="Path to the input CSV file.")
    parser.add_argument(
        "--encoding",
        "-e",
        help="CSV text encoding (examples: utf-8, utf-8-sig, cp1252, latin1, utf-16).",
    )
    parser.add_argument(
        "--sep",
        help="CSV delimiter (comma, semicolon, tab, etc). If omitted, the script will try to auto-detect.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to write the cleaned CSV (default: cleaned_carrot_prices_for_ML.csv next to this script).",
    )
    args = parser.parse_args()

    input_path = _resolve_input_csv_path(args.input)
    output_path = Path(args.output).expanduser() if args.output else (Path(__file__).resolve().parent / "cleaned_carrot_prices_for_ML.csv")

    # 1. Load the dataset
    df = _read_csv_with_encoding_fallback(input_path, args.encoding, args.sep)

# ==========================================
# STEP 1: DATA CLEANING
# ==========================================
# Rename columns to fix typos and remove broken characters
    df = df.rename(columns={
        'Temperature (蚓)': 'Temperature_C',
        'vegitable_Commodity': 'Vegetable_Commodity',
        'vegitable_Price per Unit (LKR/kg)': 'Vegetable_Price_LKR_kg',
        'fruit_Commodity': 'Fruit_Commodity',
        'fruit_Price per Unit (LKR/kg)': 'Fruit_Price_LKR_kg',
        'Crop Yield Impact Score': 'Crop_Yield_Impact_Score',
        'Rainfall (mm)': 'Rainfall_mm',
        'Humidity (%)': 'Humidity_pct'
    })

# Force the 'Crop Yield Impact Score' to become numeric. 
# The 'coerce' command turns the hidden "s" text into a blank NaN (Not a Number) value
    df['Crop_Yield_Impact_Score'] = pd.to_numeric(df['Crop_Yield_Impact_Score'], errors='coerce')

# Impute (fill) those new blank values with the median score of the dataset
    df['Crop_Yield_Impact_Score'] = df['Crop_Yield_Impact_Score'].fillna(df['Crop_Yield_Impact_Score'].median())

# ==========================================
# STEP 2: FILTERING & FEATURE ENGINEERING
# ==========================================
# Filter the massive dataset to focus solely on predicting Carrot prices
    df_carrot = df[df['Vegetable_Commodity'] == 'Carrot'].copy()

# Convert the raw string 'Date' into a proper datetime format
    df_carrot['Date'] = pd.to_datetime(df_carrot['Date'])

# Extract specific time features so the ML model can learn seasonal market trends
    df_carrot['Month'] = df_carrot['Date'].dt.month
    df_carrot['Day_of_Week'] = df_carrot['Date'].dt.dayofweek
    df_carrot['Year'] = df_carrot['Date'].dt.year

# Drop the columns we don't need for predicting vegetable prices
    df_carrot = df_carrot.drop(columns=['Fruit_Commodity', 'Fruit_Price_LKR_kg', 'Vegetable_Commodity'])

# ==========================================
# STEP 3: TIME-SERIES WINDOWING (Lag Features)
# ==========================================
# Sort the data chronologically by Region and Date
    df_carrot = df_carrot.sort_values(by=['Region', 'Date'])

# Create a "Lag Feature" (The sliding window technique we discussed)
# This adds a column telling the model what the exact price of Carrots was 7 days ago in that region
    df_carrot['Price_7_Days_Ago'] = df_carrot.groupby('Region')['Vegetable_Price_LKR_kg'].shift(7)

# Drop the first 7 days of rows since they won't have a "7 days ago" history yet
    df_carrot = df_carrot.dropna()

# ==========================================
# STEP 4: ENCODING CATEGORICAL DATA
# ==========================================
# Machine learning models don't understand text like "Colombo" or "Kandy".
# We use One-Hot Encoding (get_dummies) to turn the 'Region' column into binary 1s and 0s.
    df_carrot = pd.get_dummies(df_carrot, columns=['Region'], drop_first=True)

    # Save the final, machine-learning-ready dataset!
    df_carrot.to_csv(output_path, index=False)
    print(f"Wrote cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()