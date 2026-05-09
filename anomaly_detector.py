"""
Anomaly Detector for Journal Entry Data
=========================================
Automatically detects anomalies in raw GL/journal entry CSVs.
Supports BOTH single-line format (one row per JE with debit+credit columns)
AND paired-line format (two rows per JE, e.g. EntryNo 1.1 / 1.2).

Detection rules: duplicates (with pair matching), post-close entries,
round-dollar amounts, unusual account pairings, and excessive amounts.
"""

import re
import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Column mapping — handles various CSV column naming conventions
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    "entry_id": [
        "entry_id", "je_id", "id", "entry_number", "journal_id",
        "transaction_id", "entryno", "entry_no", "entry no", "je_no",
        "jeno", "je no", "doc_no", "voucher_no", "voucher",
    ],
    "date": [
        "date", "post_date", "posting_date", "entry_date",
        "transaction_date", "je_date", "doc_date", "effective_date",
    ],
    "account_debit": [
        "account_debit", "debit_account", "debit_acct", "dr_account", "debit",
    ],
    "account_credit": [
        "account_credit", "credit_account", "credit_acct", "cr_account", "credit",
    ],
    "account": [
        "account_key", "account", "acct", "account_code", "acct_code",
        "gl_account", "gl_code", "account_no", "account_number", "acct_no",
    ],
    "amount": [
        "amount", "entry_amount", "total", "value", "debit_amount",
        "transaction_amount", "net_amount", "amt",
    ],
    "posting_user": [
        "posting_user", "user", "posted_by", "preparer", "created_by",
        "user_id", "username", "entered_by",
    ],
    "description": [
        "description", "desc", "memo", "narrative", "comments",
        "entry_description", "detail", "details", "particulars",
        "line_description", "text",
    ],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various column names to our standard names."""
    col_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_lower)

    mapping = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and standard_name not in mapping.values():
                mapping[alias] = standard_name
                break

    df = df.rename(columns=mapping)
    return df


# ---------------------------------------------------------------------------
# Amount parsing — handles (1,234), -1234, $1,234.56 etc.
# ---------------------------------------------------------------------------

def parse_amount(val) -> float:
    """Parse amount from various string formats."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return 0.0

    # Detect parenthesized negatives: (884) → -884
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        negative = True
        s = s[1:].strip()

    # Remove currency symbols and whitespace
    s = re.sub(r"[$ £€¥,\s]", "", s)

    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return 0.0


def clean_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the amount column into proper floats."""
    if "amount" in df.columns:
        df["amount"] = df["amount"].apply(parse_amount)
    return df


# ---------------------------------------------------------------------------
# Paired-line detection & conversion
# ---------------------------------------------------------------------------

def is_paired_line_format(df: pd.DataFrame) -> bool:
    """
    Detect if the data uses paired-line JE format.
    Indicators: entry_id has decimal sub-numbers (1.1, 1.2) OR
    there is a single 'account' column instead of debit/credit columns.
    """
    has_separate_accts = "account_debit" in df.columns and "account_credit" in df.columns

    if has_separate_accts:
        return False

    # Check for sub-numbered entry IDs (e.g. "1.1", "1.2")
    if "entry_id" in df.columns:
        sample = df["entry_id"].astype(str).head(20)
        decimal_count = sample.str.match(r"^\d+\.\d+$").sum()
        if decimal_count > len(sample) * 0.3:
            return True

    # Has a generic 'account' column but no debit/credit split
    if "account" in df.columns and not has_separate_accts:
        return True

    return False


def convert_paired_to_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert paired-line JE format to single-line format.
    E.g. rows 1.1 & 1.2 become one row with account_debit and account_credit.
    """
    df = df.copy()

    # Determine the base entry number
    if "entry_id" in df.columns:
        ids = df["entry_id"].astype(str)
        if ids.str.match(r"^\d+\.\d+$").sum() > len(ids) * 0.3:
            # Sub-numbered: extract the integer part
            df["_base"] = ids.str.extract(r"^(\d+)")[0]
        else:
            # No sub-numbers — try grouping consecutive pairs
            df["_base"] = (df.index // 2).astype(str)
    else:
        df["_base"] = (df.index // 2).astype(str)

    acct_col = "account" if "account" in df.columns else None
    rows = []

    for base_id, group in df.groupby("_base", sort=False):
        if len(group) < 2:
            # Single-line entry, keep as-is with best effort
            r = group.iloc[0]
            rows.append({
                "entry_id": f"JE-{base_id}",
                "date": r.get("date", ""),
                "account_debit": str(r.get(acct_col, "")) if acct_col else str(r.get("account_debit", "")),
                "account_credit": "",
                "amount": abs(float(r.get("amount", 0))),
                "posting_user": r.get("posting_user", ""),
                "description": r.get("description", ""),
            })
            continue

        # Split into positive (debit) and negative (credit) lines
        amounts = group["amount"].values.astype(float)
        pos_mask = amounts >= 0
        neg_mask = amounts < 0

        if pos_mask.any() and neg_mask.any():
            debit_idx = np.where(pos_mask)[0][0]
            credit_idx = np.where(neg_mask)[0][0]
        else:
            # Same sign — just use first two rows
            debit_idx, credit_idx = 0, 1

        dr = group.iloc[debit_idx]
        cr = group.iloc[credit_idx]

        debit_acct = str(dr[acct_col]) if acct_col else str(dr.get("account_debit", ""))
        credit_acct = str(cr[acct_col]) if acct_col else str(cr.get("account_credit", ""))

        rows.append({
            "entry_id": f"JE-{base_id}",
            "date": dr.get("date", ""),
            "account_debit": debit_acct,
            "account_credit": credit_acct,
            "amount": abs(float(dr.get("amount", 0))),
            "posting_user": dr.get("posting_user", ""),
            "description": dr.get("description", ""),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Account code extraction
# ---------------------------------------------------------------------------

def extract_account_code(account_str: str) -> str:
    """Extract numeric account code from strings like '5400 - Office Supplies'."""
    if pd.isna(account_str):
        return ""
    s = str(account_str).strip()
    for sep in [" - ", " – ", "-", "_", " "]:
        if sep in s:
            code = s.split(sep)[0].strip()
            if code.isdigit():
                return code
    return s


# ---------------------------------------------------------------------------
# Chart of Accounts (COA) parsing
# ---------------------------------------------------------------------------

# Keywords used to auto-detect account category from COA description/type columns
_CATEGORY_KEYWORDS = {
    "asset": ["asset", "cash", "bank", "receivable", "inventory", "prepaid",
              "equipment", "property", "intangible", "fixed asset", "investment"],
    "liability": ["liability", "payable", "accrued", "loan", "debt", "mortgage",
                  "deferred revenue", "unearned", "provision", "obligation"],
    "equity": ["equity", "capital", "retained", "share", "stock", "reserve",
               "owner", "partner", "dividend", "drawing"],
    "revenue": ["revenue", "income", "sales", "fee income", "service income",
                "interest income", "gain"],
    "expense": ["expense", "cost", "salary", "wage", "rent", "depreciation",
                "amortization", "insurance", "utilities", "supplies", "loss",
                "cogs", "cost of sales", "cost of goods"],
}


def parse_coa(coa_df: pd.DataFrame) -> dict:
    """
    Parse a Chart of Accounts file into {account_code: category} mapping.
    Accepts various column formats. Tries to find an account code column
    and a type/category/description column to auto-classify.
    """
    coa_df = coa_df.copy()
    coa_df.columns = [c.lower().strip() for c in coa_df.columns]

    # Find the account code column
    code_col = None
    for candidate in ["account_key", "account_code", "acct_code", "account",
                       "code", "acct", "account_no", "account_number", "gl_code",
                       "no", "number", "id", "key"]:
        if candidate in coa_df.columns:
            code_col = candidate
            break
    if code_col is None:
        # Use first column as fallback
        code_col = coa_df.columns[0]

    # Find the category/type column (explicit classification)
    type_col = None
    for candidate in ["type", "account_type", "acct_type", "category",
                       "classification", "class", "group", "account_group",
                       "account_category", "statement"]:
        if candidate in coa_df.columns:
            type_col = candidate
            break

    # Find description/name column (for keyword-based inference)
    desc_col = None
    for candidate in ["description", "name", "account_name", "acct_name",
                       "detail", "details", "label", "title", "heading",
                       "sub_group", "subgroup", "sub_category"]:
        if candidate in coa_df.columns:
            desc_col = candidate
            break

    account_map = {}

    for _, row in coa_df.iterrows():
        code = str(row[code_col]).strip()
        # Normalize code — extract digits
        code_clean = code.split(".")[0].split("-")[0].strip()
        if not code_clean:
            continue

        category = "unknown"

        # Method 1: Use explicit type column
        if type_col:
            raw_type = str(row[type_col]).lower().strip()
            for cat, keywords in _CATEGORY_KEYWORDS.items():
                if any(kw in raw_type for kw in keywords):
                    category = cat
                    break
            # Direct match for common labels
            if category == "unknown":
                direct = {"a": "asset", "l": "liability", "e": "equity",
                          "r": "revenue", "i": "revenue", "x": "expense",
                          "bs": "asset", "pl": "expense", "is": "expense",
                          "balance sheet": "asset", "profit and loss": "expense",
                          "income statement": "expense"}
                for key, val in direct.items():
                    if raw_type == key or raw_type.startswith(key + " "):
                        category = val
                        break

        # Method 2: Infer from description column
        if category == "unknown" and desc_col:
            raw_desc = str(row[desc_col]).lower().strip()
            for cat, keywords in _CATEGORY_KEYWORDS.items():
                if any(kw in raw_desc for kw in keywords):
                    category = cat
                    break

        account_map[code_clean] = category

    return account_map


def get_account_category(code: str, account_map: Optional[dict] = None) -> str:
    """
    Determine account category from code.
    If an account_map (from uploaded COA) is provided, use it.
    Otherwise fall back to the standard 4-digit convention (1xxx-5xxx).
    """
    if not code:
        return "unknown"

    code_clean = str(code).strip().split(".")[0].split("-")[0].strip()

    # Priority 1: User-provided COA mapping
    if account_map:
        if code_clean in account_map:
            return account_map[code_clean]

    # Priority 2: Standard 4-digit convention
    if code_clean.isdigit():
        num = int(code_clean)
        if num >= 1000:
            first = int(code_clean[0])
            categories = {1: "asset", 2: "liability", 3: "equity", 4: "revenue", 5: "expense"}
            return categories.get(first, "unknown")

    return "unknown"


# ---------------------------------------------------------------------------
# Anomaly detection rules
# ---------------------------------------------------------------------------

def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Find potential duplicate entries (same date + amount + accounts)."""
    flags = []
    if not all(c in df.columns for c in ["date", "amount"]):
        return pd.DataFrame(flags)

    # Create a working copy with stringified columns for safe groupby
    work = df.copy()
    work["_date_str"] = work["date"].astype(str)
    work["_amt_round"] = work["amount"].apply(lambda x: round(float(x), 2))

    group_cols = ["_date_str", "_amt_round"]
    if "account_debit" in work.columns:
        work["_dr"] = work["account_debit"].astype(str)
        group_cols.append("_dr")
    if "account_credit" in work.columns:
        work["_cr"] = work["account_credit"].astype(str)
        group_cols.append("_cr")
    if "description" in work.columns and len(group_cols) < 4:
        work["_desc"] = work["description"].astype(str)
        group_cols.append("_desc")

    grouped = work.groupby(group_cols)

    for key, group in grouped:
        if len(group) > 1:
            entry_ids = df.loc[group.index, "entry_id"].tolist() if "entry_id" in df.columns else group.index.tolist()
            for idx in group.index:
                row = df.loc[idx]
                other_ids = [eid for eid in entry_ids if eid != row.get("entry_id", idx)]
                flags.append({
                    "index": idx,
                    "flag_type": "Duplicate Entry",
                    "flag_reason": (
                        f"Potential duplicate: same amount (${float(row['amount']):,.2f}), "
                        f"date ({row['date']}), and accounts. "
                        f"Matching entries: {', '.join(str(x) for x in other_ids)}"
                    ),
                    "severity": "high",
                    "related_entries": other_ids,
                })

    return pd.DataFrame(flags)


def detect_post_close(df: pd.DataFrame, close_date: str = "2026-01-03",
                      approved_users: Optional[list] = None) -> pd.DataFrame:
    """Find entries posted after the period-end close date."""
    flags = []
    if "date" not in df.columns:
        return pd.DataFrame(flags)

    if approved_users is None:
        approved_users = ["jsmith", "mwilliams"]

    try:
        close_dt = pd.to_datetime(close_date)
    except Exception:
        return pd.DataFrame(flags)

    for idx, row in df.iterrows():
        try:
            entry_date = pd.to_datetime(row["date"], dayfirst=True)
        except Exception:
            continue

        if entry_date > close_dt:
            user = str(row.get("posting_user", "")).lower().strip()
            reason = f"Posted after close date ({close_date})"
            if user and user not in [u.lower() for u in approved_users]:
                reason += f"; user '{row.get('posting_user', 'unknown')}' not on approved post-close list"
                severity = "high"
            else:
                severity = "medium"

            flags.append({
                "index": idx,
                "flag_type": "Post-Close Entry",
                "flag_reason": reason,
                "severity": severity,
                "related_entries": [],
            })

    return pd.DataFrame(flags)


def detect_round_dollar(df: pd.DataFrame, threshold: float = 50000) -> pd.DataFrame:
    """Find entries with suspiciously round dollar amounts above threshold."""
    flags = []
    if "amount" not in df.columns:
        return pd.DataFrame(flags)

    for idx, row in df.iterrows():
        try:
            amount = float(row["amount"])
        except (ValueError, TypeError):
            continue

        if amount >= threshold and amount % 1000 == 0:
            flags.append({
                "index": idx,
                "flag_type": "Round-Dollar Amount",
                "flag_reason": f"Round-dollar amount (${amount:,.0f}) above ${threshold:,.0f} threshold",
                "severity": "medium",
                "related_entries": [],
            })

    return pd.DataFrame(flags)


def detect_unusual_pairings(df: pd.DataFrame, account_map: Optional[dict] = None) -> pd.DataFrame:
    """Find entries with unusual account pairings."""
    flags = []
    if not all(c in df.columns for c in ["account_debit", "account_credit"]):
        return pd.DataFrame(flags)

    unusual_rules = [
        ("revenue", "asset", "Revenue debited against asset account"),
        ("revenue", "expense", "Revenue debited against expense account"),
        ("equity", "expense", "Equity debited against expense account"),
        ("liability", "asset", "Liability debited against non-cash asset"),
    ]

    for idx, row in df.iterrows():
        debit_code = extract_account_code(str(row["account_debit"]))
        credit_code = extract_account_code(str(row["account_credit"]))
        debit_cat = get_account_category(debit_code, account_map)
        credit_cat = get_account_category(credit_code, account_map)

        # Skip if either category is unknown — not enough info to judge
        if debit_cat == "unknown" or credit_cat == "unknown":
            continue

        # Skip normal cash payments
        if credit_code in ("1010", "10") and debit_cat == "liability":
            continue

        for rule_debit, rule_credit, desc in unusual_rules:
            if debit_cat == rule_debit and credit_cat == rule_credit:
                flags.append({
                    "index": idx,
                    "flag_type": "Unusual Account Pairing",
                    "flag_reason": (
                        f"Unusual account combination: {row['account_debit']} ({debit_cat}) "
                        f"paired with {row['account_credit']} ({credit_cat}). {desc}."
                    ),
                    "severity": "high",
                    "related_entries": [],
                })
                break

    return pd.DataFrame(flags)


def detect_excessive_amounts(df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """Find entries with amounts far exceeding the account's average."""
    flags = []
    if not all(c in df.columns for c in ["amount", "account_debit"]):
        return pd.DataFrame(flags)

    df_calc = df.copy()
    df_calc["_amount"] = pd.to_numeric(df_calc["amount"], errors="coerce")
    account_stats = df_calc.groupby("account_debit")["_amount"].agg(["mean", "std", "count"])

    for idx, row in df.iterrows():
        try:
            amount = float(row["amount"])
        except (ValueError, TypeError):
            continue

        acct = row["account_debit"]
        if acct in account_stats.index:
            stats = account_stats.loc[acct]
            avg = stats["mean"]
            count = stats["count"]

            if count >= 3 and avg > 0 and amount > avg * multiplier:
                ratio = amount / avg
                flags.append({
                    "index": idx,
                    "flag_type": "Excessive Amount",
                    "flag_reason": (
                        f"Amount (${amount:,.0f}) exceeds {ratio:.0f}x the average "
                        f"(${avg:,.0f}) for account {acct}"
                    ),
                    "severity": "high" if ratio > 10 else "medium",
                    "related_entries": [],
                })

    return pd.DataFrame(flags)


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def run_all_detections(
    df: pd.DataFrame,
    close_date: str = "2026-01-03",
    approved_users: Optional[list] = None,
    round_threshold: float = 50000,
    excessive_multiplier: float = 3.0,
    account_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run all anomaly detection rules on a DataFrame.
    Automatically detects single-line vs paired-line JE format.
    If account_map is provided (from uploaded COA), uses it for account categorization.
    Returns flagged rows with added columns: flag_type, flag_reason, severity, related_entries.
    """
    # Normalize columns
    df = normalize_columns(df.copy())

    # Parse amounts (handles parentheses, commas, currency symbols)
    df = clean_amounts(df)

    # Ensure entry_id exists
    if "entry_id" not in df.columns:
        df["entry_id"] = [f"JE-{i+1:04d}" for i in range(len(df))]

    # Detect and convert paired-line format
    if is_paired_line_format(df):
        df = convert_paired_to_single(df)

    # Parse dates (try dayfirst for DD/MM/YY format)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Run all detectors
    all_flags = []
    detectors = [
        ("Duplicates", lambda: detect_duplicates(df)),
        ("Post-Close", lambda: detect_post_close(df, close_date, approved_users)),
        ("Round-Dollar", lambda: detect_round_dollar(df, round_threshold)),
        ("Unusual Pairings", lambda: detect_unusual_pairings(df, account_map)),
        ("Excessive Amounts", lambda: detect_excessive_amounts(df, excessive_multiplier)),
    ]

    for name, detector in detectors:
        result = detector()
        if len(result) > 0:
            all_flags.append(result)

    if not all_flags:
        return pd.DataFrame()

    flags_df = pd.concat(all_flags, ignore_index=True)

    # Merge flags back into original data
    combined_flags = flags_df.groupby("index").agg({
        "flag_type": lambda x: " | ".join(sorted(set(x))),
        "flag_reason": lambda x: " | ".join(x),
        "severity": lambda x: "high" if "high" in x.values else "medium",
        "related_entries": lambda x: [item for sublist in x for item in sublist],
    }).reset_index()

    result = df.loc[combined_flags["index"]].copy()
    result["flag_type"] = combined_flags["flag_type"].values
    result["flag_reason"] = combined_flags["flag_reason"].values
    result["severity"] = combined_flags["severity"].values
    result["related_entries"] = combined_flags["related_entries"].values

    return result.reset_index(drop=True)
