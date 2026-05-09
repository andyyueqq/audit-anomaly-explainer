"""
Synthetic Journal Entry Anomaly Generator
==========================================
Generates a CSV of flagged journal entry anomalies for the Audit Anomaly Explainer project.

Produces 5 anomaly types:
  1. Post-close entries (posted after period-end close date)
  2. Round-dollar entries (suspiciously round amounts above threshold)
  3. Unusual account pairings (debit/credit combos that rarely occur together)
  4. Duplicate entries (same amount, date, and account appearing twice)
  5. Excessive amounts (entries far exceeding the monthly average for that account)

Also includes ~30 normal (non-anomalous) entries for realism.

Output: flagged_anomalies.csv (anomalies only) + full_journal.csv (all entries)
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERIOD_END = datetime(2025, 12, 31)
PERIOD_START = datetime(2025, 12, 1)
CLOSE_DATE = datetime(2026, 1, 3)  # books closed on Jan 3

APPROVED_POST_CLOSE_USERS = ["jsmith", "mwilliams"]
ALL_USERS = ["jsmith", "mwilliams", "abrown", "klee", "jgarcia", "tcheng", "rjones", "lpatel"]

# Account chart (code -> name)
ACCOUNTS = {
    "1010": "Cash",
    "1200": "Accounts Receivable",
    "1300": "Prepaid Expenses",
    "1400": "Inventory",
    "1500": "Fixed Assets",
    "2010": "Accounts Payable",
    "2200": "Accrued Liabilities",
    "2300": "Deferred Revenue",
    "3000": "Retained Earnings",
    "4000": "Revenue",
    "4100": "Service Revenue",
    "5010": "Cost of Goods Sold",
    "5100": "Salaries Expense",
    "5200": "Rent Expense",
    "5300": "Utilities Expense",
    "5400": "Office Supplies",
    "5500": "Travel Expense",
    "5600": "Consulting Expense",
    "5700": "Depreciation Expense",
    "5800": "Miscellaneous Expense",
    "5900": "Insurance Expense",
}

# Normal debit-credit pairings (common in real GL)
NORMAL_PAIRINGS = [
    ("5100", "1010"),   # Salaries -> Cash
    ("5200", "1010"),   # Rent -> Cash
    ("5300", "2010"),   # Utilities -> AP
    ("5400", "1010"),   # Supplies -> Cash
    ("5500", "2200"),   # Travel -> Accrued
    ("1010", "4000"),   # Cash -> Revenue
    ("1200", "4100"),   # AR -> Service Revenue
    ("5010", "1400"),   # COGS -> Inventory
    ("5700", "1500"),   # Depreciation -> Fixed Assets
    ("5900", "1300"),   # Insurance -> Prepaid
    ("1300", "1010"),   # Prepaid -> Cash
    ("2010", "1010"),   # AP -> Cash (payment)
]

# Unusual pairings (would be flagged)
UNUSUAL_PAIRINGS = [
    ("4000", "1300"),   # Revenue debited against Prepaid (odd)
    ("3000", "5800"),   # Retained Earnings debited against Misc Expense
    ("1500", "4000"),   # Fixed Assets debited against Revenue
    ("2300", "1400"),   # Deferred Revenue debited against Inventory
]

entry_id_counter = 1000


def next_id():
    global entry_id_counter
    entry_id_counter += 1
    return f"JE-{entry_id_counter}"


def random_date_in_period():
    days = (PERIOD_END - PERIOD_START).days
    return PERIOD_START + timedelta(days=random.randint(0, days))


def random_post_close_date():
    return CLOSE_DATE + timedelta(days=random.randint(1, 10))


def fmt_date(d):
    return d.strftime("%Y-%m-%d")


def make_entry(date, debit_code, credit_code, amount, user, description, flag_reason=None):
    return {
        "entry_id": next_id(),
        "date": fmt_date(date),
        "account_debit": f"{debit_code} - {ACCOUNTS[debit_code]}",
        "account_credit": f"{credit_code} - {ACCOUNTS[credit_code]}",
        "amount": round(amount, 2),
        "posting_user": user,
        "description": description,
        "flag_reason": flag_reason or "",
    }


# ---------------------------------------------------------------------------
# Generate normal entries
# ---------------------------------------------------------------------------
normal_entries = []
for _ in range(30):
    debit, credit = random.choice(NORMAL_PAIRINGS)
    amount = round(random.uniform(500, 25000), 2)
    user = random.choice(ALL_USERS)
    date = random_date_in_period()
    desc_templates = [
        f"Monthly {ACCOUNTS[debit].lower()} recording",
        f"Period-end {ACCOUNTS[debit].lower()} accrual",
        f"Payment for {ACCOUNTS[debit].lower()}",
        f"Standard {ACCOUNTS[debit].lower()} entry",
        f"Recurring {ACCOUNTS[debit].lower()} transaction",
    ]
    normal_entries.append(make_entry(date, debit, credit, amount, user, random.choice(desc_templates)))


# ---------------------------------------------------------------------------
# Generate anomalies (4 per type = 20 flagged entries)
# ---------------------------------------------------------------------------
anomalies = []

# --- Type 1: Post-close entries ---
for i in range(4):
    debit, credit = random.choice(NORMAL_PAIRINGS)
    amount = round(random.uniform(5000, 80000), 2)
    # 2 by approved users, 2 by unapproved
    if i < 2:
        user = random.choice([u for u in ALL_USERS if u not in APPROVED_POST_CLOSE_USERS])
    else:
        user = random.choice(APPROVED_POST_CLOSE_USERS)
    date = random_post_close_date()
    desc = random.choice([
        "Late adjustment to period-end accruals",
        "Correction entry for prior period",
        "Post-close reclassification",
        "Year-end true-up adjustment",
    ])
    flag = f"Posted after close date ({fmt_date(CLOSE_DATE)})"
    if user not in APPROVED_POST_CLOSE_USERS:
        flag += f"; user '{user}' not on approved post-close list"
    anomalies.append(make_entry(date, debit, credit, amount, user, desc, flag))

# --- Type 2: Round-dollar entries ---
round_amounts = [50000, 100000, 75000, 200000]
for i, amt in enumerate(round_amounts):
    debit, credit = random.choice(NORMAL_PAIRINGS[:6])
    user = random.choice(ALL_USERS)
    date = random_date_in_period()
    desc = random.choice([
        "Consulting services - Q4 engagement",
        "Annual software license renewal",
        "Marketing campaign payment",
        "Strategic advisory fee",
    ])
    anomalies.append(make_entry(date, debit, credit, float(amt), user, desc,
                                f"Round-dollar amount (${amt:,.0f}) above $50K threshold"))

# --- Type 3: Unusual account pairings ---
for i, (debit, credit) in enumerate(UNUSUAL_PAIRINGS):
    amount = round(random.uniform(3000, 50000), 2)
    user = random.choice(ALL_USERS)
    date = random_date_in_period()
    desc = random.choice([
        "Reclassification per management request",
        "Adjustment entry - see supporting memo",
        "Correction of prior posting error",
        "Manual adjustment per controller",
    ])
    anomalies.append(make_entry(date, debit, credit, amount, user, desc,
                                f"Unusual account combination: {ACCOUNTS[debit]} paired with {ACCOUNTS[credit]}"))

# --- Type 4: Duplicate entries ---
for i in range(2):  # 2 pairs = 4 entries
    debit, credit = random.choice(NORMAL_PAIRINGS)
    amount = round(random.uniform(2000, 20000), 2)
    user = random.choice(ALL_USERS)
    date = random_date_in_period()
    desc = f"Vendor payment - invoice #{random.randint(10000, 99999)}"
    flag = f"Potential duplicate: same amount (${amount:,.2f}), date ({fmt_date(date)}), and accounts"
    # Create the pair
    anomalies.append(make_entry(date, debit, credit, amount, user, desc, flag))
    anomalies.append(make_entry(date, debit, credit, amount, user, desc, flag))

# --- Type 5: Excessive amounts ---
excessive_accounts = [
    ("5400", "1010", "Office Supplies", 5000, 500000),    # avg 5K, entry 500K
    ("5500", "2200", "Travel Expense", 8000, 120000),     # avg 8K, entry 120K
    ("5300", "2010", "Utilities Expense", 3000, 85000),   # avg 3K, entry 85K
    ("5800", "1010", "Miscellaneous Expense", 2000, 60000),  # avg 2K, entry 60K
]
for debit, credit, name, avg, amount in excessive_accounts:
    user = random.choice(ALL_USERS)
    date = random_date_in_period()
    ratio = amount / avg
    desc = random.choice([
        f"Bulk purchase - {name.lower()}",
        f"Annual {name.lower()} contract",
        f"One-time {name.lower()} expenditure",
        f"Special project - {name.lower()}",
    ])
    anomalies.append(make_entry(date, debit, credit, float(amount), user, desc,
                                f"Amount (${amount:,.0f}) exceeds {ratio:.0f}x monthly average (${avg:,.0f}) for {name}"))


# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------
FIELDS = ["entry_id", "date", "account_debit", "account_credit", "amount",
          "posting_user", "description", "flag_reason"]

output_dir = os.path.dirname(os.path.abspath(__file__))

# Full journal (normal + anomalies, shuffled)
all_entries = normal_entries + anomalies
random.shuffle(all_entries)

full_path = os.path.join(output_dir, "full_journal.csv")
with open(full_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(all_entries)

# Flagged anomalies only
flagged = [e for e in all_entries if e["flag_reason"]]
flagged_path = os.path.join(output_dir, "flagged_anomalies.csv")
with open(flagged_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(flagged)

print(f"Generated {len(all_entries)} total journal entries ({len(normal_entries)} normal, {len(flagged)} flagged)")
print(f"  -> {full_path}")
print(f"  -> {flagged_path}")
print()
print("Anomaly breakdown:")
types = {}
for e in flagged:
    reason = e["flag_reason"]
    if "after close" in reason.lower():
        t = "Post-close"
    elif "round-dollar" in reason.lower():
        t = "Round-dollar"
    elif "unusual account" in reason.lower():
        t = "Unusual pairing"
    elif "duplicate" in reason.lower():
        t = "Duplicate"
    elif "exceeds" in reason.lower():
        t = "Excessive amount"
    else:
        t = "Other"
    types[t] = types.get(t, 0) + 1
for t, c in sorted(types.items()):
    print(f"  {t}: {c}")
