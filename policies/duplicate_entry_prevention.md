# Duplicate Entry Prevention and Detection Policy

**Policy Number:** FIN-DP-004
**Effective Date:** January 1, 2025
**Last Revised:** September 15, 2024
**Owner:** Accounting Manager

## 1. Purpose

This policy establishes procedures for preventing, detecting, and resolving duplicate journal entries in the general ledger. Duplicate entries can lead to material misstatements in financial reports and may indicate control weaknesses.

## 2. Definition of Duplicate Entry

A journal entry is considered a potential duplicate if it matches another entry on two or more of the following criteria within the same accounting period:

- **Same date** of posting.
- **Same dollar amount** (to the cent).
- **Same account codes** (both debit and credit accounts).
- **Same or similar description** (determined by text similarity analysis).
- **Same posting user.**

An exact match on date, amount, and accounts is classified as a **high-confidence duplicate** and requires immediate investigation. A match on only two of the five criteria is classified as a **possible duplicate** and is flagged for review.

## 3. Prevention Controls

### 3.1 System Controls

The ERP system is configured with the following duplicate-prevention controls:

- **Warning on identical entries:** If a user attempts to post an entry with the same date, amount, and account codes as an existing entry, the system will display a warning and require the user to confirm the posting.
- **Batch duplicate check:** Before any batch upload of journal entries, the system runs a duplicate scan against the current period. Identified duplicates are quarantined for review before posting.

### 3.2 Process Controls

- Preparers must search for existing entries before posting a new entry with the same account and approximate amount.
- Recurring journal entries should use the auto-reversing or recurring-entry module rather than manual re-entry each period to minimize the risk of double-posting.

## 4. Detection Procedures

### 4.1 Weekly Scan

The data-analytics team runs a weekly duplicate-detection script (Alteryx workflow) that identifies:

- Exact duplicates (same date, amount, debit account, credit account).
- Near-duplicates (same amount and at least one matching account, posted within 3 business days).

Results are distributed to the Accounting Manager and the Internal Audit team every Monday.

### 4.2 Monthly Review

During the monthly close process, the Senior Accountant reviews the duplicate-detection report and confirms resolution status for all flagged items.

## 5. Resolution Procedures

When a potential duplicate is identified:

1. **Investigation:** The preparer of the later entry must confirm whether the entry is a true duplicate or a legitimate separate transaction.
2. **Documentation:** If the entry is legitimate, the preparer must add a note to the entry description explaining why it is not a duplicate (e.g., "Second payment per contract amendment dated 2025-03-15").
3. **Reversal:** If the entry is confirmed as a duplicate, it must be reversed within 2 business days. The reversal entry must reference the original duplicate entry ID.
4. **Root cause:** If duplicate entries occur more than twice in a quarter from the same preparer or process, the Accounting Manager must conduct a root-cause analysis and implement corrective action.

## 6. Reporting

Duplicate entry statistics (count of flagged items, confirmed duplicates, reversal timeliness) are reported monthly to the Controller and quarterly to Internal Audit as part of the financial controls dashboard.

## 7. Exceptions

Certain transaction types commonly produce legitimate entries with identical amounts and accounts within the same period. These include:

- Bi-weekly payroll entries (same gross payroll amount for consecutive pay periods).
- Monthly rent payments to the same landlord.
- Recurring utility payments.

These known patterns are excluded from the weekly duplicate scan through a maintained exception list, reviewed and updated quarterly by the Accounting Manager.
