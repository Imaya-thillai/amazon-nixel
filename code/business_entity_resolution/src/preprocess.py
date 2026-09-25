"""Data Preprocessing & Text Normalization Module.

Handles cross-lingual normalization for US, India, and France:
- Unicode NFKD decomposition (stripping accents and diacritics)
- Legal entity suffix canonicalization and root name extraction
- Street and unit abbreviation expansion
- Postal/PIN code and numeric token extraction
- Robust, fault-tolerant TSV parsing
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


# Legal entity suffix patterns across US, India, France
LEGAL_SUFFIXES = [
    # US / International
    r"\bincorporated\b", r"\binc\b",
    r"\bcorporation\b", r"\bcorp\b",
    r"\blimited liability company\b", r"\bllc\b", r"\bl\.l\.c\b",
    r"\bcompany\b", r"\bco\b",
    r"\bpublic limited company\b", r"\bplc\b",
    # India
    r"\bprivate limited\b", r"\bpvt ltd\b", r"\bpvt\b", r"\bltd\b", r"\blimited\b",
    r"\benterprises\b", r"\btraders\b", r"\band sons\b", r"\band co\b",
    # France
    r"\bsarl\b", r"\bsas\b", r"\bs\.a\b", r"\beurl\b",
    r"\bsociete\b", r"\bets\b", r"\betablissements\b", r"\bcie\b", r"\bcompagnie\b"
]

LEGAL_SUFFIX_REGEX = re.compile(r"|".join(LEGAL_SUFFIXES), re.IGNORECASE)

# Address abbreviation mappings
ADDRESS_ABBREVIATIONS = {
    # US / General
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bsq\b": "square",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bp\.?o\.?\s*box\b": "pobox",
    # India
    r"\bngr\b": "nagar",
    r"\bcol\b": "colony",
    r"\bext\b": "extension",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\brly\b": "railway",
    r"\bstn\b": "station",
    r"\bsec\b": "sector",
    r"\bdist\b": "district",
    # France
    r"\br\.\b": "rue",
    r"\bbd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\bpl\b": "place",
    r"\ball\b": "allee",
    r"\bimp\b": "impasse",
    r"\bchem\b": "chemin"
}

POSTAL_CODE_REGEX = re.compile(r"\b(\d{5,6})\b")


def strip_accents(text: str) -> str:
    """Normalize unicode and strip accent diacritics (e.g. é -> e, ô -> o)."""
    if not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_text(text: str) -> str:
    """Basic lowercasing, accent stripping, and whitespace normalization."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text.lower())
    text = text.replace("&", " and ")
    # Replace non-alphanumeric (except hyphen) with spaces
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_root_name(cleaned_name: str) -> str:
    """Removes legal entity designations to isolate core brand/root name."""
    root = LEGAL_SUFFIX_REGEX.sub(" ", cleaned_name)
    return re.sub(r"\s+", " ", root).strip()


def normalize_address(address: str) -> str:
    """Normalizes address string, expanding common abbreviations."""
    addr = clean_text(address)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        addr = re.sub(pattern, replacement, addr)
    return re.sub(r"\s+", " ", addr).strip()


def extract_postal_code(address: str) -> Optional[str]:
    """Extracts 5-digit (US/France) or 6-digit (India) postal code if present."""
    match = POSTAL_CODE_REGEX.search(address)
    return match.group(1) if match else None


def extract_digits(text: str) -> List[str]:
    """Extracts all contiguous digit sequences from text (house numbers, postal codes)."""
    return re.findall(r"\b\d+\b", text)


def load_source_tsv(
    filepath: str,
    max_rows: Optional[int] = None,
    target_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Robust TSV parser for source records.

    Handles noisy columns, unescaped quotes, and inconsistent tab separators safely.
    Supports max_rows and target_ids filtering for low-RAM streaming.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")

        for line_num, line in enumerate(f, start=2):
            if max_rows and len(rows) >= max_rows:
                break
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            entity_id = parts[0].strip()

            if target_ids is not None and entity_id not in target_ids:
                continue

            if len(parts) == 4:
                rows.append([entity_id, parts[1].strip(), parts[2].strip(), parts[3].strip()])
            elif len(parts) > 4:
                country = parts[-1].strip()
                middle = " ".join(p.strip() for p in parts[1:-1])
                sub_parts = middle.split("  ", 1)
                b_name = sub_parts[0] if sub_parts else middle
                b_addr = sub_parts[1] if len(sub_parts) > 1 else ""
                rows.append([entity_id, b_name, b_addr, country])
            elif len(parts) == 3:
                rows.append([entity_id, parts[1].strip(), parts[2].strip(), ""])
            else:
                rows.append([entity_id, "", "", ""])

    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    return preprocess_dataframe(df)


def load_ground_truth_tsv(filepath: str) -> Dict[str, List[str]]:
    """Loads train_ground_truth.tsv into a clean mapping: source1_id -> list of matched IDs."""
    mapping = {}
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)  # Skip header
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            s1, _, rest = line.partition("\t")
            s1 = s1.strip()
            if not s1:
                continue
            matched = [m.strip() for m in rest.split(",") if m.strip()] if rest.strip() else []
            mapping[s1] = matched
    return mapping


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies complete normalization pipeline to a source DataFrame."""
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str).str.strip().str.upper()

    df["clean_name"] = df["business_name"].apply(clean_text)
    df["root_name"] = df["clean_name"].apply(extract_root_name)
    df["clean_address"] = df["business_address"].apply(normalize_address)
    df["postal_code"] = df["clean_address"].apply(extract_postal_code)
    df["combined_text"] = df["clean_name"] + " " + df["clean_address"]

    return df
