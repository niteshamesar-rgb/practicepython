import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

APP_TITLE = "Personal Finance Dashboard"
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
TXN_CSV_PATH = DATA_DIR / "transactions.csv"
RULES_JSON_PATH = DATA_DIR / "rules.json"

DEFAULT_CATEGORIES = [
    "Income",
    "Rent",
    "Groceries",
    "Dining",
    "Transport",
    "Utilities",
    "Entertainment",
    "Shopping",
    "Healthcare",
    "Subscriptions",
    "Travel",
    "Taxes",
    "Savings",
    "Transfers",
    "Other",
]

DEFAULT_RULES = {
    "rules": [
        {"pattern": "uber|lyft|taxi|ride", "category": "Transport"},
        {"pattern": "starbucks|coffee|cafe", "category": "Dining"},
        {"pattern": "walmart|costco|aldi|kroger|grocery", "category": "Groceries"},
        {"pattern": "netflix|spotify|prime|subscription", "category": "Subscriptions"},
        {"pattern": "rent|mortgage", "category": "Rent"},
        {"pattern": "electric|water|gas|utility", "category": "Utilities"},
        {"pattern": "flight|airbnb|hotel|booking", "category": "Travel"},
        {"pattern": "pharmacy|hospital|clinic|doctor|dental", "category": "Healthcare"},
        {"pattern": "salary|payroll|paycheck|bonus", "category": "Income"},
    ]
}

REQUIRED_COLUMNS = [
    "date",
    "description",
    "amount",
    "type",
    "category",
    "account",
    "tags",
    "notes",
]


def ensure_data_dir() -> None:
    """Ensure the data directory and baseline files exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TXN_CSV_PATH.exists():
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        empty_df.to_csv(TXN_CSV_PATH, index=False)
    if not RULES_JSON_PATH.exists():
        RULES_JSON_PATH.write_text(json.dumps(DEFAULT_RULES, indent=2))


def load_rules() -> Dict[str, List[Dict[str, str]]]:
    if not RULES_JSON_PATH.exists():
        return DEFAULT_RULES
    try:
        return json.loads(RULES_JSON_PATH.read_text())
    except Exception:
        return DEFAULT_RULES


def save_rules(rules: Dict[str, List[Dict[str, str]]]) -> None:
    RULES_JSON_PATH.write_text(json.dumps(rules, indent=2))


def load_transactions() -> pd.DataFrame:
    if not TXN_CSV_PATH.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    try:
        df = pd.read_csv(TXN_CSV_PATH)
    except Exception:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description"] = df["description"].fillna("")
    df["type"] = df["type"].fillna("")
    df["category"] = df["category"].fillna("Other")
    df["account"] = df["account"].fillna("Main")
    df["tags"] = df["tags"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df


def save_transactions(df: pd.DataFrame) -> None:
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_out.to_csv(TXN_CSV_PATH, index=False)


def apply_rules(df: pd.DataFrame, rules: Dict[str, List[Dict[str, str]]]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["description_lower"] = df["description"].str.lower().fillna("")
    for rule in rules.get("rules", []):
        pattern = rule.get("pattern", "").lower()
        category = rule.get("category", "Other")
        if not pattern:
            continue
        mask = df["description_lower"].str.contains(pattern, regex=True, na=False)
        df.loc[mask & df["category"].isin(["", "Other", np.nan]), "category"] = category
    df.drop(columns=["description_lower"], inplace=True)
    return df


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["date"]).sort_values("date")

    # Standardize type
    def normalize_type(x: str) -> str:
        if not isinstance(x, str):
            return "expense" if np.nan else "expense"
        x_lower = x.strip().lower()
        if x_lower in {"inc", "income", "+", "credit"}:
            return "income"
        if x_lower in {"exp", "expense", "-", "debit"}:
            return "expense"
        if x_lower in {"transfer", "xfer"}:
            return "transfer"
        return "expense"

    df["type"] = df["type"].apply(normalize_type)

    # Normalize amount sign: expenses negative, income positive
    def normalize_amount(row: pd.Series) -> float:
        amt = row.get("amount", np.nan)
        if pd.isna(amt):
            return np.nan
        if row.get("type") == "expense" and amt > 0:
            return -float(amt)
        if row.get("type") == "income" and amt < 0:
            return -float(amt)
        return float(amt)

    df["amount"] = df.apply(normalize_amount, axis=1)

    # Derive month for grouping
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Ensure categories set
    df["category"] = df["category"].replace({"": "Other", np.nan: "Other"})

    return df


def add_transaction_form(existing_categories: List[str]) -> Optional[pd.DataFrame]:
    st.subheader("Add Transaction")
    with st.form("add_txn_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            date_val = st.date_input("Date")
            account = st.text_input("Account", value="Main")
            txn_type = st.selectbox("Type", ["expense", "income", "transfer"], index=0)
        with col2:
            description = st.text_input("Description")
            amount = st.number_input("Amount", step=0.01, format="%.2f")
            category = st.selectbox("Category", options=sorted(set(DEFAULT_CATEGORIES + existing_categories)))
        with col3:
            tags = st.text_input("Tags (comma-separated)")
            notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Add")

    if not submitted:
        return None

    new_row = {
        "date": pd.to_datetime(date_val),
        "description": description,
        "amount": float(amount),
        "type": txn_type,
        "category": category,
        "account": account,
        "tags": tags,
        "notes": notes,
    }
    return pd.DataFrame([new_row], columns=REQUIRED_COLUMNS)


def bulk_import_uploader() -> Optional[pd.DataFrame]:
    st.subheader("Bulk Import")
    st.caption("Upload CSV or Excel with columns: date, description, amount, type, category, account, tags, notes")
    file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if file is None:
        return None

    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    column_mapping = {c: c for c in REQUIRED_COLUMNS}

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.warning("Some required columns are missing. Map them below.")
        options = ["<skip>"] + list(df.columns)
        map_cols = {}
        for col in REQUIRED_COLUMNS:
            default_idx = options.index(col) if col in options else 0
            sel = st.selectbox(f"Map for '{col}'", options=options, index=default_idx, key=f"map_{col}")
            map_cols[col] = None if sel == "<skip>" else sel

        new_df = pd.DataFrame()
        for col in REQUIRED_COLUMNS:
            src = map_cols.get(col)
            if src is not None and src in df.columns:
                new_df[col] = df[src]
            else:
                new_df[col] = np.nan
        df = new_df
    else:
        df = df[REQUIRED_COLUMNS]

    # Coerce types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description"] = df["description"].astype(str)
    df["type"] = df["type"].astype(str)
    df["category"] = df["category"].astype(str)
    df["account"] = df["account"].astype(str)
    df["tags"] = df["tags"].astype(str)
    df["notes"] = df["notes"].astype(str)

    st.success(f"Imported {len(df)} rows. Review and click 'Append to Ledger' to save.")
    st.dataframe(df.head(50), use_container_width=True)

    if st.button("Append to Ledger"):
        return df
    return None


def rules_manager(rules: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    st.subheader("Categorization Rules")
    st.caption("Rules apply to the description using regex. First match wins for uncategorized items.")

    existing = rules.get("rules", [])
    for idx, rule in enumerate(existing):
        with st.expander(f"Rule {idx + 1}"):
            pattern = st.text_input("Pattern (regex)", value=rule.get("pattern", ""), key=f"pat_{idx}")
            category = st.text_input("Category", value=rule.get("category", "Other"), key=f"cat_{idx}")
            if st.button("Delete", key=f"del_{idx}"):
                existing.pop(idx)
                break
            else:
                rule["pattern"] = pattern
                rule["category"] = category

    st.markdown("---")
    st.markdown("Add New Rule")
    new_pattern = st.text_input("New Pattern", key="new_pat")
    new_category = st.text_input("New Category", key="new_cat")
    if st.button("Add Rule") and new_pattern and new_category:
        existing.append({"pattern": new_pattern, "category": new_category})

    return {"rules": existing}


def render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard")
    if df.empty:
        st.info("No transactions yet. Add or import to see analytics.")
        return

    min_date = df["date"].min()
    max_date = df["date"].max()

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()))
        with c2:
            selected_accounts = st.multiselect("Accounts", options=sorted(df["account"].dropna().unique().tolist()))
        with c3:
            selected_cats = st.multiselect("Categories", options=sorted(df["category"].dropna().unique().tolist()))

    fdf = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        fdf = fdf[(fdf["date"] >= start_date) & (fdf["date"] <= end_date)]
    if selected_accounts:
        fdf = fdf[fdf["account"].isin(selected_accounts)]
    if selected_cats:
        fdf = fdf[fdf["category"].isin(selected_cats)]

    income_total = fdf.loc[fdf["type"] == "income", "amount"].sum()
    expense_total = fdf.loc[fdf["type"] == "expense", "amount"].sum()
    net_total = income_total + expense_total
    savings_rate = (net_total / income_total * 100.0) if income_total != 0 else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Income", f"${income_total:,.2f}")
    m2.metric("Expenses", f"${-expense_total:,.2f}")
    m3.metric("Net", f"${net_total:,.2f}")
    m4.metric("Savings Rate", f"{savings_rate:.1f}%" if not np.isnan(savings_rate) else "-")

    col1, col2 = st.columns(2)
    with col1:
        monthly = fdf.groupby(["year_month", "type"], as_index=False)["amount"].sum()
        monthly_pivot = monthly.pivot(index="year_month", columns="type", values="amount").fillna(0)
        monthly_pivot = monthly_pivot.reset_index()
        fig = px.bar(monthly_pivot, x="year_month", y=["income", "expense"], title="Monthly Income vs Expenses", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_cat = fdf[fdf["type"] == "expense"].groupby("category", as_index=False)["amount"].sum()
        by_cat["amount"] = by_cat["amount"].abs()
        fig2 = px.pie(by_cat, names="category", values="amount", title="Spending by Category")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.caption("Transactions")
    st.dataframe(fdf.sort_values("date", ascending=False), use_container_width=True, height=420)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    ensure_data_dir()

    rules = load_rules()
    df = load_transactions()

    # Apply rules to uncategorized rows
    df_with_rules = apply_rules(df, rules)
    if not df_with_rules.equals(df):
        save_transactions(df_with_rules)
        df = df_with_rules

    df = preprocess_transactions(df)

    with st.sidebar:
        st.header("Menu")
        page = st.radio("Go to", options=["Dashboard", "Add Transaction", "Bulk Import", "Rules", "Data"], index=0)
        st.markdown("---")
        st.caption("Data location: " + str(TXN_CSV_PATH))

    if page == "Add Transaction":
        new_df = add_transaction_form(existing_categories=sorted(df["category"].dropna().unique().tolist()))
        if new_df is not None:
            new_df = preprocess_transactions(new_df)
            combined = pd.concat([df, new_df], ignore_index=True)
            combined = apply_rules(combined, rules)
            save_transactions(combined)
            st.success("Transaction added.")
            st.experimental_rerun()

    elif page == "Bulk Import":
        imported = bulk_import_uploader()
        if imported is not None:
            imported = preprocess_transactions(imported)
            combined = pd.concat([df, imported], ignore_index=True)
            combined = apply_rules(combined, rules)
            save_transactions(combined)
            st.success("Imported transactions appended.")
            st.experimental_rerun()

    elif page == "Rules":
        updated_rules = rules_manager(rules)
        if st.button("Save Rules"):
            save_rules(updated_rules)
            st.success("Rules saved.")
            st.experimental_rerun()

    elif page == "Data":
        st.subheader("Ledger")
        st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Download CSV"):
                st.download_button("Download", data=pd.read_csv(TXN_CSV_PATH).to_csv(index=False), file_name="transactions.csv", mime="text/csv")
        with c2:
            if st.button("Clear All Data"):
                save_transactions(pd.DataFrame(columns=REQUIRED_COLUMNS))
                st.warning("All transactions cleared.")
                st.experimental_rerun()
        with c3:
            st.write("")

    else:
        render_dashboard(df)


if __name__ == "__main__":
    main()