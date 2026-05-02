"""
data_engine.py
Pure business logic — no Streamlit dependencies.
All data loading, validation, aggregation, and context building lives here.
Fully testable with pytest.
"""

import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads .env file from the project folder automatically

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_FILES = {
    "account_dim":    "account_dim.csv",
    "date_dim":       "date_dim.csv",
    "hcp_dim":        "hcp_dim.csv",
    "rep_dim":        "rep_dim.csv",
    "territory_dim":  "territory_dim.csv",
    "fact_rx":        "fact_rx.csv",
    "fact_rep_act":   "fact_rep_activity.csv",
    "fact_ln":        "fact_ln_metrics.csv",
    "fact_payor":     "fact_payor_mix.csv",
}

REQUIRED_COLUMNS = {
    "account_dim":   ["account_id", "name", "account_type", "territory_id"],
    "date_dim":      ["date_id", "calendar_date", "year", "quarter"],
    "hcp_dim":       ["hcp_id", "full_name", "specialty", "tier", "territory_id"],
    "rep_dim":       ["rep_id", "first_name", "last_name", "region"],
    "territory_dim": ["territory_id", "name"],
    "fact_rx":       ["hcp_id", "date_id", "brand_code", "trx_cnt", "nrx_cnt"],
    "fact_rep_act":  ["activity_id", "rep_id", "hcp_id", "account_id", "date_id",
                      "activity_type", "status", "duration_min"],
    "fact_ln":       ["entity_type", "entity_id", "quarter_id",
                      "ln_patient_cnt", "est_market_share"],
    "fact_payor":    ["account_id", "date_id", "payor_type", "pct_of_volume"],
}

GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all required CSV files from data_dir.

    Raises:
        ValueError:       if data_dir is empty/None, or a CSV is empty,
                          or required columns are missing.
        FileNotFoundError: if data_dir does not exist or a CSV is missing.
    """
    if not data_dir or not data_dir.strip():
        raise ValueError("data_dir must be a non-empty string.")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    data: dict[str, pd.DataFrame] = {}

    for key, fname in REQUIRED_FILES.items():
        fpath = os.path.join(data_dir, fname)

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file missing: {fpath}")

        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            raise ValueError(f"Cannot parse {fname}: {exc}") from exc

        if df.empty:
            raise ValueError(f"File is empty (zero rows): {fname}")

        missing_cols = [c for c in REQUIRED_COLUMNS[key] if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{fname} is missing required columns: {missing_cols}"
            )

        data[key] = df

    return data


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _safe_pct(numerator: float, denominator: float, decimals: int = 1) -> float:
    """Return percentage; 0.0 if denominator is zero."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, decimals)


def _safe_div(numerator: float, denominator: float, decimals: int = 1) -> float:
    """Return division result; 0.0 if denominator is zero."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, decimals)


def compute_brand_summary(fact_rx: pd.DataFrame) -> pd.DataFrame:
    """TRx / NRx totals and NRx% per brand."""
    if fact_rx.empty:
        return pd.DataFrame(columns=["brand_code", "trx_cnt", "nrx_cnt", "nrx_pct"])
    summary = fact_rx.groupby("brand_code")[["trx_cnt", "nrx_cnt"]].sum().reset_index()
    summary["nrx_pct"] = summary.apply(
        lambda r: _safe_pct(r["nrx_cnt"], r["trx_cnt"]), axis=1
    )
    return summary.sort_values("trx_cnt", ascending=False).reset_index(drop=True)


def compute_quarterly_trend(fact_rx: pd.DataFrame, date_dim: pd.DataFrame) -> pd.DataFrame:
    """TRx / NRx grouped by quarter and brand, with QoQ growth."""
    if fact_rx.empty or date_dim.empty:
        return pd.DataFrame()
    rx_dated = fact_rx.merge(date_dim[["date_id", "quarter"]], on="date_id", how="left")
    bq = (
        rx_dated.groupby(["quarter", "brand_code"])[["trx_cnt", "nrx_cnt"]]
        .sum()
        .reset_index()
        .sort_values(["brand_code", "quarter"])
    )
    bq["trx_qoq_pct"] = (
        bq.groupby("brand_code")["trx_cnt"]
        .pct_change()
        .mul(100)
        .round(1)
    )
    bq["nrx_qoq_pct"] = (
        bq.groupby("brand_code")["nrx_cnt"]
        .pct_change()
        .mul(100)
        .round(1)
    )
    return bq


def compute_rep_scorecard(
    fact_rep_act: pd.DataFrame, rep_dim: pd.DataFrame, hcp_dim: pd.DataFrame
) -> pd.DataFrame:
    """Per-rep: activities, unique HCPs, completion rate, Tier A coverage."""
    if fact_rep_act.empty or rep_dim.empty:
        return pd.DataFrame()

    named = fact_rep_act.merge(rep_dim, on="rep_id", how="left")
    named["rep_name"] = named["first_name"].fillna("") + " " + named["last_name"].fillna("")
    named["rep_name"] = named["rep_name"].str.strip()

    total   = named.groupby("rep_name")["activity_id"].count().rename("total_activities")
    unique  = named.groupby("rep_name")["hcp_id"].nunique().rename("unique_hcps")
    done    = (
        named[named["status"] == "completed"]
        .groupby("rep_name")["activity_id"]
        .count()
        .rename("completed")
    )

    scorecard = pd.concat([total, unique, done], axis=1).fillna(0)
    scorecard["completion_rate"] = scorecard.apply(
        lambda r: _safe_pct(r["completed"], r["total_activities"]), axis=1
    )
    scorecard["avg_calls_per_hcp"] = scorecard.apply(
        lambda r: _safe_div(r["total_activities"], r["unique_hcps"]), axis=1
    )

    # Tier A coverage
    if not hcp_dim.empty and "tier" in hcp_dim.columns:
        tier_merged = named.merge(hcp_dim[["hcp_id", "tier"]], on="hcp_id", how="left")
        tier_a = (
            tier_merged[tier_merged["tier"] == "A"]
            .groupby("rep_name")["activity_id"]
            .count()
            .rename("tier_a_calls")
        )
        scorecard = scorecard.join(tier_a).fillna(0)
        scorecard["tier_a_pct"] = scorecard.apply(
            lambda r: _safe_pct(r["tier_a_calls"], r["total_activities"]), axis=1
        )
    else:
        scorecard["tier_a_calls"] = 0
        scorecard["tier_a_pct"] = 0.0

    return scorecard.sort_values("total_activities", ascending=False)


def compute_territory_scorecard(
    fact_rx: pd.DataFrame,
    fact_rep_act: pd.DataFrame,
    hcp_dim: pd.DataFrame,
    rep_dim: pd.DataFrame,
    territory_dim: pd.DataFrame,
    account_dim: pd.DataFrame,
) -> pd.DataFrame:
    """Per-territory: TRx, NRx, activities, HCP count, efficiency ratios."""
    if territory_dim.empty:
        return pd.DataFrame()

    terr_names = territory_dim.set_index("territory_id")["name"].to_dict()

    # Rx via hcp
    rx_terr = pd.DataFrame()
    if not fact_rx.empty and not hcp_dim.empty:
        rx_terr = (
            fact_rx.merge(hcp_dim[["hcp_id", "territory_id"]], on="hcp_id", how="left")
            .merge(territory_dim[["territory_id", "name"]], on="territory_id", how="left")
            .groupby("name")[["trx_cnt", "nrx_cnt"]]
            .sum()
        )

    # Activities via rep region
    act_terr = pd.Series(dtype=int, name="activity_count")
    if not fact_rep_act.empty and not rep_dim.empty:
        act_terr = (
            fact_rep_act.merge(rep_dim[["rep_id", "region"]], on="rep_id", how="left")
            .groupby("region")["activity_id"]
            .count()
            .rename("activity_count")
        )

    # HCP count
    hcp_terr = pd.Series(dtype=int, name="hcp_count")
    if not hcp_dim.empty:
        hcp_terr = (
            hcp_dim.merge(territory_dim[["territory_id", "name"]], on="territory_id", how="left")
            .groupby("name")["hcp_id"]
            .count()
            .rename("hcp_count")
        )

    summary = rx_terr.join(act_terr, how="outer").join(hcp_terr, how="outer").fillna(0)
    summary["trx_per_hcp"] = summary.apply(
        lambda r: _safe_div(r["trx_cnt"], r["hcp_count"]), axis=1
    )
    summary["act_per_hcp"] = summary.apply(
        lambda r: _safe_div(r["activity_count"], r["hcp_count"]), axis=1
    )
    return summary


def compute_hcp_coverage(
    fact_rep_act: pd.DataFrame, hcp_dim: pd.DataFrame
) -> dict:
    """Returns coverage stats and list of never-called HCPs."""
    all_ids    = set(hcp_dim["hcp_id"].tolist()) if not hcp_dim.empty else set()
    called_ids = set(fact_rep_act["hcp_id"].tolist()) if not fact_rep_act.empty else set()
    never      = all_ids - called_ids

    never_df = pd.DataFrame()
    if never and not hcp_dim.empty:
        never_df = hcp_dim[hcp_dim["hcp_id"].isin(never)][
            ["full_name", "tier", "specialty"]
        ].copy()

    return {
        "total_hcps":    len(all_ids),
        "called_hcps":   len(called_ids),
        "never_called":  len(never),
        "coverage_pct":  _safe_pct(len(called_ids), len(all_ids)),
        "never_called_df": never_df,
    }


def compute_payor_summary(
    fact_payor: pd.DataFrame, account_dim: pd.DataFrame, date_dim: pd.DataFrame
) -> dict:
    """Overall and per-account payor averages."""
    if fact_payor.empty:
        return {"overall": pd.Series(dtype=float), "by_account": pd.DataFrame()}

    overall = fact_payor.groupby("payor_type")["pct_of_volume"].mean().round(1)

    by_account = pd.DataFrame()
    if not account_dim.empty:
        pa = fact_payor.merge(
            account_dim[["account_id", "name"]], on="account_id", how="left"
        )
        by_account = (
            pa.groupby(["name", "payor_type"])["pct_of_volume"]
            .mean()
            .unstack(fill_value=0)
            .round(1)
        )

    return {"overall": overall, "by_account": by_account}


def compute_market_share(fact_ln: pd.DataFrame) -> dict:
    """Overall and quarterly average market share."""
    if fact_ln.empty:
        return {"overall_avg": 0.0, "overall_patient_avg": 0.0, "by_quarter": pd.Series(dtype=float)}

    return {
        "overall_avg":         round(fact_ln["est_market_share"].mean(), 1),
        "overall_patient_avg": round(fact_ln["ln_patient_cnt"].mean(), 1),
        "by_quarter": (
            fact_ln.groupby("quarter_id")["est_market_share"]
            .mean()
            .round(1)
            .sort_index()
        ),
    }


# ── Insights engine ───────────────────────────────────────────────────────────

def compute_insights(
    rep_scorecard: pd.DataFrame,
    territory_scorecard: pd.DataFrame,
    quarterly_trend: pd.DataFrame,
    hcp_coverage: dict,
    payor_summary: dict,
) -> list[str]:
    """
    Derive flagged business insights from pre-computed aggregations.
    Returns a list of plain-English insight strings with severity tags.
    """
    insights = []

    # ── Rep Tier A coverage flags
    if not rep_scorecard.empty and "tier_a_pct" in rep_scorecard.columns:
        avg_tier_a = rep_scorecard["tier_a_pct"].mean()
        for rep, row in rep_scorecard.iterrows():
            pct = row["tier_a_pct"]
            if pct < avg_tier_a * 0.6:
                insights.append(
                    f"[ALERT] {rep} has critically low Tier A coverage: "
                    f"{pct}% vs team avg {avg_tier_a:.1f}%."
                )
            elif pct < avg_tier_a * 0.85:
                insights.append(
                    f"[WARN] {rep} is below average on Tier A coverage: "
                    f"{pct}% vs team avg {avg_tier_a:.1f}%."
                )

    # ── Rep completion rate flags
    if not rep_scorecard.empty and "completion_rate" in rep_scorecard.columns:
        avg_comp = rep_scorecard["completion_rate"].mean()
        for rep, row in rep_scorecard.iterrows():
            if row["completion_rate"] < 60:
                insights.append(
                    f"[ALERT] {rep} has a low call completion rate: "
                    f"{row['completion_rate']}% (team avg {avg_comp:.1f}%)."
                )

    # ── Territory efficiency flags
    if not territory_scorecard.empty and "trx_per_hcp" in territory_scorecard.columns:
        avg_eff = territory_scorecard["trx_per_hcp"].mean()
        for terr, row in territory_scorecard.iterrows():
            if row["trx_per_hcp"] < avg_eff * 0.75:
                insights.append(
                    f"[ALERT] {terr} has low TRx efficiency: "
                    f"{row['trx_per_hcp']} TRx/HCP vs avg {avg_eff:.1f}."
                )

    # ── NRx declining trend flag
    if not quarterly_trend.empty and "nrx_qoq_pct" in quarterly_trend.columns:
        last_q = quarterly_trend.groupby("brand_code").last().reset_index()
        for _, row in last_q.iterrows():
            if pd.notna(row["nrx_qoq_pct"]) and row["nrx_qoq_pct"] < -10:
                insights.append(
                    f"[ALERT] {row['brand_code']} NRx declined {row['nrx_qoq_pct']}% "
                    f"in the most recent quarter — new patient acquisition is falling."
                )
            elif pd.notna(row["nrx_qoq_pct"]) and row["nrx_qoq_pct"] > 10:
                insights.append(
                    f"[POSITIVE] {row['brand_code']} NRx grew {row['nrx_qoq_pct']}% "
                    f"in the most recent quarter."
                )

    # ── HCP coverage gap flags
    never_df = hcp_coverage.get("never_called_df", pd.DataFrame())
    if not never_df.empty:
        tier_a_uncalled = never_df[never_df["tier"] == "A"]
        if not tier_a_uncalled.empty:
            names = ", ".join(tier_a_uncalled["full_name"].tolist())
            insights.append(
                f"[CRITICAL] {len(tier_a_uncalled)} Tier A HCP(s) have NEVER been called: {names}."
            )
        tier_b_uncalled = never_df[never_df["tier"] == "B"]
        if not tier_b_uncalled.empty:
            insights.append(
                f"[WARN] {len(tier_b_uncalled)} Tier B HCP(s) have never been called."
            )

    # ── High Medicare payor risk
    by_account = payor_summary.get("by_account", pd.DataFrame())
    if not by_account.empty and "Medicare" in by_account.columns:
        high_medicare = by_account[by_account["Medicare"] > 60]
        for acct in high_medicare.index:
            insights.append(
                f"[WARN] {acct} has high Medicare exposure: "
                f"{high_medicare.loc[acct, 'Medicare']}% — reimbursement risk."
            )

    if not insights:
        insights.append("[INFO] No significant anomalies detected in the current data.")

    return insights


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(data: dict[str, pd.DataFrame]) -> str:
    """
    Compute all aggregations and return a rich, structured context string
    for the LLM system prompt.
    """
    account_dim   = data["account_dim"]
    date_dim      = data["date_dim"]
    hcp_dim       = data["hcp_dim"]
    rep_dim       = data["rep_dim"]
    territory_dim = data["territory_dim"]
    fact_rx       = data["fact_rx"]
    fact_rep_act  = data["fact_rep_act"]
    fact_ln       = data["fact_ln"]
    fact_payor    = data["fact_payor"]

    brand_summary    = compute_brand_summary(fact_rx)
    quarterly_trend  = compute_quarterly_trend(fact_rx, date_dim)
    rep_scorecard    = compute_rep_scorecard(fact_rep_act, rep_dim, hcp_dim)
    territory_sc     = compute_territory_scorecard(
        fact_rx, fact_rep_act, hcp_dim, rep_dim, territory_dim, account_dim
    )
    hcp_coverage     = compute_hcp_coverage(fact_rep_act, hcp_dim)
    payor_summary    = compute_payor_summary(fact_payor, account_dim, date_dim)
    market_share     = compute_market_share(fact_ln)
    insights         = compute_insights(
        rep_scorecard, territory_sc, quarterly_trend, hcp_coverage, payor_summary
    )

    lines = []

    # ── Dataset overview
    lines.append("=== DATASET OVERVIEW ===")
    lines.append(
        f"Territories: {len(territory_dim)} | Reps: {len(rep_dim)} | "
        f"HCPs: {len(hcp_dim)} | Accounts: {len(account_dim)}"
    )
    if not date_dim.empty:
        lines.append(
            f"Date range: {date_dim['calendar_date'].min()} to {date_dim['calendar_date'].max()}"
        )
    lines.append(
        f"Total Rx records: {len(fact_rx):,} | "
        f"Total activity records: {len(fact_rep_act):,}"
    )

    # ── Flagged insights (LLM reads these first)
    lines.append("\n=== FLAGGED INSIGHTS & ALERTS ===")
    for ins in insights:
        lines.append(f"  {ins}")

    # ── Brands
    lines.append("\n=== BRANDS — TRx / NRx TOTALS ===")
    for _, row in brand_summary.iterrows():
        lines.append(
            f"  {row['brand_code']}: TRx={int(row['trx_cnt']):,}  "
            f"NRx={int(row['nrx_cnt']):,}  NRx%={row['nrx_pct']}%"
        )

    # ── Quarterly trend
    lines.append("\n=== QUARTERLY RX TREND BY BRAND (with QoQ growth) ===")
    for _, row in quarterly_trend.iterrows():
        qoq_trx = f"{row['trx_qoq_pct']:+.1f}%" if pd.notna(row["trx_qoq_pct"]) else "N/A"
        qoq_nrx = f"{row['nrx_qoq_pct']:+.1f}%" if pd.notna(row["nrx_qoq_pct"]) else "N/A"
        lines.append(
            f"  {row['quarter']} | {row['brand_code']}: "
            f"TRx={int(row['trx_cnt']):,} ({qoq_trx} QoQ)  "
            f"NRx={int(row['nrx_cnt']):,} ({qoq_nrx} QoQ)"
        )

    # ── HCP tiers
    lines.append("\n=== HCP TIER BREAKDOWN ===")
    tier_counts = hcp_dim["tier"].value_counts().sort_index() if not hcp_dim.empty else {}
    for tier, cnt in tier_counts.items():
        lines.append(f"  Tier {tier}: {cnt} HCPs")

    # ── Rx by HCP tier
    lines.append("\n=== RX PERFORMANCE BY HCP TIER ===")
    if not fact_rx.empty and not hcp_dim.empty:
        rx_hcp = fact_rx.merge(hcp_dim[["hcp_id", "tier"]], on="hcp_id", how="left")
        tier_rx = (
            rx_hcp.groupby("tier")[["trx_cnt", "nrx_cnt"]]
            .sum()
            .reindex(["A", "B", "C"])
            .fillna(0)
        )
        total_trx = tier_rx["trx_cnt"].sum()
        for tier, row in tier_rx.iterrows():
            share = _safe_pct(row["trx_cnt"], total_trx)
            lines.append(
                f"  Tier {tier}: TRx={int(row['trx_cnt']):,} ({share}% of total)  "
                f"NRx={int(row['nrx_cnt']):,}"
            )

    # ── Rep scorecard
    lines.append("\n=== REP SCORECARD ===")
    for rep, row in rep_scorecard.iterrows():
        tier_a_info = ""
        if "tier_a_pct" in row:
            tier_a_info = f"  Tier_A_calls={int(row['tier_a_calls'])} ({row['tier_a_pct']}%)"
        lines.append(
            f"  {rep}: activities={int(row['total_activities'])}  "
            f"unique_HCPs={int(row['unique_hcps'])}  "
            f"completion={row['completion_rate']}%  "
            f"avg_calls/HCP={row['avg_calls_per_hcp']}"
            f"{tier_a_info}"
        )

    # ── Activity type & status
    lines.append("\n=== ACTIVITY TYPE MIX ===")
    if not fact_rep_act.empty:
        for atype, cnt in fact_rep_act["activity_type"].value_counts().items():
            lines.append(f"  {atype}: {cnt} ({_safe_pct(cnt, len(fact_rep_act))}%)")

    lines.append("\n=== ACTIVITY STATUS ===")
    if not fact_rep_act.empty:
        for status, cnt in fact_rep_act["status"].value_counts().items():
            lines.append(f"  {status}: {cnt} ({_safe_pct(cnt, len(fact_rep_act))}%)")

    # ── HCP coverage
    lines.append("\n=== HCP CALL COVERAGE ===")
    lines.append(
        f"  Called at least once: {hcp_coverage['called_hcps']} / "
        f"{hcp_coverage['total_hcps']} ({hcp_coverage['coverage_pct']}%)"
    )
    lines.append(f"  Never called: {hcp_coverage['never_called']}")
    never_df = hcp_coverage["never_called_df"]
    if not never_df.empty:
        for _, row in never_df.iterrows():
            lines.append(
                f"    - {row['full_name']} (Tier {row['tier']}, {row['specialty']})"
            )

    # ── Territory scorecard
    lines.append("\n=== TERRITORY SCORECARD ===")
    for terr, row in territory_sc.iterrows():
        lines.append(
            f"  {terr}: TRx={int(row.get('trx_cnt', 0)):,}  "
            f"NRx={int(row.get('nrx_cnt', 0)):,}  "
            f"HCPs={int(row.get('hcp_count', 0))}  "
            f"Activities={int(row.get('activity_count', 0))}  "
            f"TRx/HCP={row.get('trx_per_hcp', 0)}  "
            f"Act/HCP={row.get('act_per_hcp', 0)}"
        )

    # ── Payor mix
    lines.append("\n=== PAYOR MIX — OVERALL AVERAGE ===")
    for payor, pct in payor_summary["overall"].items():
        lines.append(f"  {payor}: {pct}% avg")

    lines.append("\n=== PAYOR MIX — BY ACCOUNT ===")
    by_account = payor_summary["by_account"]
    if not by_account.empty:
        for acct, row in by_account.iterrows():
            parts = "  |  ".join(f"{p}: {v}%" for p, v in row.items())
            lines.append(f"  {acct}: {parts}")

    # ── Market share
    lines.append("\n=== MARKET SHARE (LN METRICS) ===")
    lines.append(f"  Overall avg market share: {market_share['overall_avg']}%")
    lines.append(f"  Overall avg LN patient count: {market_share['overall_patient_avg']}")
    for q, ms in market_share["by_quarter"].items():
        lines.append(f"  {q}: avg market share = {ms}%")

    # ── HCP specialties
    lines.append("\n=== HCP SPECIALTIES ===")
    if not hcp_dim.empty:
        for spec, cnt in hcp_dim["specialty"].value_counts().items():
            lines.append(f"  {spec}: {cnt} HCPs")

    return "\n".join(lines)


# ── Groq client ───────────────────────────────────────────────────────────────

def get_groq_client(api_key: str) -> Groq:
    """
    Validate and return a Groq client.

    Raises:
        ValueError: if api_key is empty, None, or whitespace-only.
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY is missing or empty. "
            "Set it as an environment variable before starting the app."
        )
    return Groq(api_key=api_key.strip())


# ── LLM call ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a senior pharma sales analyst AI assistant named Aria.
You have access to pre-computed analytics from a real pharmaceutical sales dataset.

RULES:
1. Answer ONLY from the data provided below. Do not invent numbers.
2. Always cite your source section (e.g. "According to REP SCORECARD..." or "From TERRITORY SCORECARD...").
3. If the data does not contain enough information to answer, say: "I don't have enough data to answer that — the dataset covers [what it does cover]."
4. When ranking or comparing, always show the actual numbers side by side.
5. End business answers with one actionable recommendation where relevant.
6. For trend questions, reference the QoQ growth numbers explicitly.

--- ANALYTICS DATA ---
{context}
--- END OF DATA ---
"""


def build_system_prompt(context: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)


def ask_question(
    client: Groq,
    system_prompt: str,
    history: list[dict],
    question: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Send a question to Groq and return the answer string.

    Raises:
        ValueError: if question is empty/None or client is None.
        RuntimeError: if the Groq API returns an unexpected response.
    """
    if client is None:
        raise ValueError("Groq client is None — call get_groq_client() first.")
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": question.strip()}
    ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Groq returned an empty response — no choices.")

    content = choices[0].message.content
    if content is None:
        raise RuntimeError("Groq returned a null message content.")

    return content
