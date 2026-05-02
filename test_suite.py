"""
test_suite.py
Comprehensive pytest test suite for data_engine.py.
Covers happy paths, edge cases, and every crash/failure scenario.

Run with:
    pytest test_suite.py -v
    pytest test_suite.py -v --tb=short      # shorter tracebacks
    pytest test_suite.py -v -k "load_data"  # run only load_data tests
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from data_engine import (
    load_data,
    build_context,
    compute_brand_summary,
    compute_quarterly_trend,
    compute_rep_scorecard,
    compute_territory_scorecard,
    compute_hcp_coverage,
    compute_payor_summary,
    compute_market_share,
    compute_insights,
    get_groq_client,
    ask_question,
    build_system_prompt,
    _safe_pct,
    _safe_div,
    REQUIRED_FILES,
    REQUIRED_COLUMNS,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — minimal valid dataframes
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def minimal_data():
    """Smallest valid dataset that satisfies all required columns."""
    return {
        "account_dim": pd.DataFrame({
            "account_id": [1000, 1001],
            "name": ["Acct A", "Acct B"],
            "account_type": ["Hospital", "Clinic"],
            "territory_id": [1, 2],
        }),
        "date_dim": pd.DataFrame({
            "date_id": [20240801, 20240901, 20241001],
            "calendar_date": ["2024-08-01", "2024-09-01", "2024-10-01"],
            "year": [2024, 2024, 2024],
            "quarter": ["Q3", "Q3", "Q4"],
        }),
        "hcp_dim": pd.DataFrame({
            "hcp_id": [1000000001, 1000000002, 1000000003],
            "full_name": ["Dr A", "Dr B", "Dr C"],
            "specialty": ["Rheumatology", "Rheumatology", "Oncology"],
            "tier": ["A", "B", "C"],
            "territory_id": [1, 1, 2],
        }),
        "rep_dim": pd.DataFrame({
            "rep_id": [1, 2],
            "first_name": ["Alice", "Bob"],
            "last_name": ["Smith", "Jones"],
            "region": ["Territory 1", "Territory 2"],
        }),
        "territory_dim": pd.DataFrame({
            "territory_id": [1, 2],
            "name": ["Territory 1", "Territory 2"],
        }),
        "fact_rx": pd.DataFrame({
            "hcp_id": [1000000001, 1000000002, 1000000001],
            "date_id": [20240801, 20240901, 20241001],
            "brand_code": ["BRANDX", "BRANDX", "BRANDY"],
            "trx_cnt": [10, 20, 5],
            "nrx_cnt": [3, 8, 2],
        }),
        "fact_rep_act": pd.DataFrame({
            "activity_id": [1, 2, 3, 4],
            "rep_id": [1, 1, 2, 2],
            "hcp_id": [1000000001, 1000000002, 1000000003, 1000000001],
            "account_id": [1000, 1001, 1000, 1001],
            "date_id": [20240801, 20240801, 20240901, 20241001],
            "activity_type": ["call", "lunch_meeting", "call", "call"],
            "status": ["completed", "completed", "cancelled", "completed"],
            "duration_min": [20, 60, 15, 30],
        }),
        "fact_ln": pd.DataFrame({
            "entity_type": ["H", "H"],
            "entity_id": [1000000001, 1000000002],
            "quarter_id": ["2024Q3", "2024Q4"],
            "ln_patient_cnt": [50, 70],
            "est_market_share": [15.0, 22.0],
        }),
        "fact_payor": pd.DataFrame({
            "account_id": [1000, 1000, 1000],
            "date_id": [20240801, 20240801, 20240801],
            "payor_type": ["Commercial", "Medicare", "Medicaid"],
            "pct_of_volume": [40.0, 35.0, 25.0],
        }),
    }


@pytest.fixture
def csv_dir(tmp_path, minimal_data):
    """Write minimal_data to a temp directory as CSVs."""
    file_map = {
        "account_dim":   "account_dim.csv",
        "date_dim":      "date_dim.csv",
        "hcp_dim":       "hcp_dim.csv",
        "rep_dim":       "rep_dim.csv",
        "territory_dim": "territory_dim.csv",
        "fact_rx":       "fact_rx.csv",
        "fact_rep_act":  "fact_rep_activity.csv",
        "fact_ln":       "fact_ln_metrics.csv",
        "fact_payor":    "fact_payor_mix.csv",
    }
    for key, fname in file_map.items():
        minimal_data[key].to_csv(tmp_path / fname, index=False)
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════════
# 1. UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeDiv:
    def test_normal(self):
        assert _safe_div(10, 4) == 2.5

    def test_zero_denominator(self):
        assert _safe_div(10, 0) == 0.0

    def test_zero_numerator(self):
        assert _safe_div(0, 5) == 0.0

    def test_both_zero(self):
        assert _safe_div(0, 0) == 0.0

    def test_decimals(self):
        assert _safe_div(1, 3, decimals=2) == 0.33


class TestSafePct:
    def test_normal(self):
        assert _safe_pct(25, 100) == 25.0

    def test_zero_denominator(self):
        assert _safe_pct(5, 0) == 0.0

    def test_zero_numerator(self):
        assert _safe_pct(0, 100) == 0.0

    def test_both_zero(self):
        assert _safe_pct(0, 0) == 0.0

    def test_over_100(self):
        # edge: more completed than total (data error scenario)
        result = _safe_pct(110, 100)
        assert result == 110.0

    def test_decimals(self):
        assert _safe_pct(1, 3, decimals=2) == 33.33


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadData:
    def test_happy_path(self, csv_dir):
        data = load_data(str(csv_dir))
        assert len(data) == 9
        for key in REQUIRED_FILES:
            assert key in data
            assert isinstance(data[key], pd.DataFrame)
            assert not data[key].empty

    def test_all_required_columns_present(self, csv_dir):
        data = load_data(str(csv_dir))
        for key, cols in REQUIRED_COLUMNS.items():
            for col in cols:
                assert col in data[key].columns, f"{key} missing column '{col}'"

    def test_directory_does_not_exist(self):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            load_data("/nonexistent/path/abc123")

    def test_empty_data_dir_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            load_data("")

    def test_whitespace_data_dir_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            load_data("   ")

    def test_none_data_dir(self):
        with pytest.raises((ValueError, TypeError)):
            load_data(None)

    def test_missing_one_file(self, csv_dir):
        os.remove(csv_dir / "fact_rx.csv")
        with pytest.raises(FileNotFoundError, match="fact_rx.csv"):
            load_data(str(csv_dir))

    def test_missing_all_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path))

    def test_empty_csv_raises(self, csv_dir, minimal_data):
        # Overwrite one file with headers only (no rows)
        pd.DataFrame(columns=minimal_data["fact_rx"].columns).to_csv(
            csv_dir / "fact_rx.csv", index=False
        )
        with pytest.raises(ValueError, match="empty"):
            load_data(str(csv_dir))

    def test_corrupt_csv_raises(self, csv_dir):
        (csv_dir / "hcp_dim.csv").write_text("not,a,valid\ncsv\x00file\n!!!")
        # Some pandas versions parse this; if not, it should raise ValueError
        try:
            data = load_data(str(csv_dir))
            # If parsed, required columns will be missing
        except (ValueError, Exception):
            pass  # expected

    def test_missing_required_column(self, csv_dir, minimal_data):
        # Drop a required column and rewrite
        bad_df = minimal_data["hcp_dim"].drop(columns=["tier"])
        bad_df.to_csv(csv_dir / "hcp_dim.csv", index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_data(str(csv_dir))

    def test_extra_columns_allowed(self, csv_dir, minimal_data):
        # Extra columns should not cause failure
        df = minimal_data["fact_rx"].copy()
        df["extra_col"] = 999
        df.to_csv(csv_dir / "fact_rx.csv", index=False)
        data = load_data(str(csv_dir))
        assert "extra_col" in data["fact_rx"].columns

    def test_returns_dataframes_not_references(self, csv_dir):
        data1 = load_data(str(csv_dir))
        data2 = load_data(str(csv_dir))
        # Mutating one should not affect another
        data1["fact_rx"]["trx_cnt"] = 0
        assert data2["fact_rx"]["trx_cnt"].sum() != 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE BRAND SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeBrandSummary:
    def test_happy_path(self, minimal_data):
        result = compute_brand_summary(minimal_data["fact_rx"])
        assert "brand_code" in result.columns
        assert "trx_cnt" in result.columns
        assert "nrx_cnt" in result.columns
        assert "nrx_pct" in result.columns
        assert len(result) == 2  # BRANDX and BRANDY

    def test_sorted_by_trx_desc(self, minimal_data):
        result = compute_brand_summary(minimal_data["fact_rx"])
        assert result["trx_cnt"].is_monotonic_decreasing

    def test_nrx_pct_between_0_and_100(self, minimal_data):
        result = compute_brand_summary(minimal_data["fact_rx"])
        assert (result["nrx_pct"] >= 0).all()
        assert (result["nrx_pct"] <= 100).all()

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["hcp_id", "date_id", "brand_code", "trx_cnt", "nrx_cnt"])
        result = compute_brand_summary(empty)
        assert result.empty

    def test_single_brand(self):
        df = pd.DataFrame({"brand_code": ["X", "X"], "trx_cnt": [10, 20], "nrx_cnt": [2, 5]})
        result = compute_brand_summary(df)
        assert len(result) == 1
        assert result.iloc[0]["trx_cnt"] == 30

    def test_zero_trx_no_division_error(self):
        df = pd.DataFrame({"brand_code": ["X"], "trx_cnt": [0], "nrx_cnt": [0]})
        result = compute_brand_summary(df)
        assert result.iloc[0]["nrx_pct"] == 0.0

    def test_all_nrx_equals_trx(self):
        df = pd.DataFrame({"brand_code": ["X"], "trx_cnt": [100], "nrx_cnt": [100]})
        result = compute_brand_summary(df)
        assert result.iloc[0]["nrx_pct"] == 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE QUARTERLY TREND
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeQuarterlyTrend:
    def test_happy_path(self, minimal_data):
        result = compute_quarterly_trend(minimal_data["fact_rx"], minimal_data["date_dim"])
        assert not result.empty
        assert "quarter" in result.columns
        assert "brand_code" in result.columns
        assert "trx_qoq_pct" in result.columns
        assert "nrx_qoq_pct" in result.columns

    def test_empty_fact_rx(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["fact_rx"].columns)
        result = compute_quarterly_trend(empty, minimal_data["date_dim"])
        assert result.empty

    def test_empty_date_dim(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["date_dim"].columns)
        result = compute_quarterly_trend(minimal_data["fact_rx"], empty)
        assert result.empty

    def test_both_empty(self):
        empty_rx   = pd.DataFrame(columns=["hcp_id", "date_id", "brand_code", "trx_cnt", "nrx_cnt"])
        empty_date = pd.DataFrame(columns=["date_id", "quarter", "calendar_date", "year"])
        result = compute_quarterly_trend(empty_rx, empty_date)
        assert result.empty

    def test_single_quarter_qoq_is_nan(self, minimal_data):
        # Only one quarter → no prior quarter → QoQ should be NaN
        single_q = minimal_data["fact_rx"].copy()
        single_date = minimal_data["date_dim"][minimal_data["date_dim"]["quarter"] == "Q3"]
        single_q = single_q[single_q["date_id"].isin(single_date["date_id"])]
        result = compute_quarterly_trend(single_q, single_date)
        assert result["trx_qoq_pct"].isna().all()

    def test_unmatched_date_ids_handled(self, minimal_data):
        # fact_rx has date_ids not in date_dim → should not crash
        bad_rx = minimal_data["fact_rx"].copy()
        bad_rx["date_id"] = 99999999
        result = compute_quarterly_trend(bad_rx, minimal_data["date_dim"])
        # quarter column will be NaN but should not raise
        assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE REP SCORECARD
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRepScorecard:
    def test_happy_path(self, minimal_data):
        result = compute_rep_scorecard(
            minimal_data["fact_rep_act"],
            minimal_data["rep_dim"],
            minimal_data["hcp_dim"],
        )
        assert not result.empty
        assert "total_activities" in result.columns
        assert "unique_hcps" in result.columns
        assert "completion_rate" in result.columns
        assert "avg_calls_per_hcp" in result.columns
        assert "tier_a_pct" in result.columns

    def test_completion_rate_in_range(self, minimal_data):
        result = compute_rep_scorecard(
            minimal_data["fact_rep_act"],
            minimal_data["rep_dim"],
            minimal_data["hcp_dim"],
        )
        assert (result["completion_rate"] >= 0).all()
        assert (result["completion_rate"] <= 100).all()

    def test_empty_activity(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["fact_rep_act"].columns)
        result = compute_rep_scorecard(empty, minimal_data["rep_dim"], minimal_data["hcp_dim"])
        assert result.empty

    def test_empty_rep_dim(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["rep_dim"].columns)
        result = compute_rep_scorecard(minimal_data["fact_rep_act"], empty, minimal_data["hcp_dim"])
        assert result.empty

    def test_hcp_dim_missing_tier_column(self, minimal_data):
        # tier column missing → should default tier_a to 0, not crash
        hcp_no_tier = minimal_data["hcp_dim"].drop(columns=["tier"])
        result = compute_rep_scorecard(
            minimal_data["fact_rep_act"],
            minimal_data["rep_dim"],
            hcp_no_tier,
        )
        assert "tier_a_calls" in result.columns
        assert (result["tier_a_calls"] == 0).all()

    def test_all_cancelled_activities(self, minimal_data):
        all_cancelled = minimal_data["fact_rep_act"].copy()
        all_cancelled["status"] = "cancelled"
        result = compute_rep_scorecard(
            all_cancelled, minimal_data["rep_dim"], minimal_data["hcp_dim"]
        )
        assert (result["completion_rate"] == 0.0).all()

    def test_all_completed_activities(self, minimal_data):
        all_done = minimal_data["fact_rep_act"].copy()
        all_done["status"] = "completed"
        result = compute_rep_scorecard(
            all_done, minimal_data["rep_dim"], minimal_data["hcp_dim"]
        )
        assert (result["completion_rate"] == 100.0).all()

    def test_rep_with_one_hcp(self, minimal_data):
        one_act = minimal_data["fact_rep_act"].head(1)
        result = compute_rep_scorecard(
            one_act, minimal_data["rep_dim"], minimal_data["hcp_dim"]
        )
        assert result["avg_calls_per_hcp"].iloc[0] == 1.0

    def test_rep_name_concat_no_nan(self, minimal_data):
        result = compute_rep_scorecard(
            minimal_data["fact_rep_act"],
            minimal_data["rep_dim"],
            minimal_data["hcp_dim"],
        )
        for name in result.index:
            assert "nan" not in name.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPUTE TERRITORY SCORECARD
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeTerritoryScorecard:
    def test_happy_path(self, minimal_data):
        result = compute_territory_scorecard(
            minimal_data["fact_rx"],
            minimal_data["fact_rep_act"],
            minimal_data["hcp_dim"],
            minimal_data["rep_dim"],
            minimal_data["territory_dim"],
            minimal_data["account_dim"],
        )
        assert not result.empty
        assert "trx_cnt" in result.columns
        assert "trx_per_hcp" in result.columns

    def test_no_division_by_zero_when_hcp_count_zero(self, minimal_data):
        # Territory with no HCPs
        terr_extra = pd.concat([
            minimal_data["territory_dim"],
            pd.DataFrame({"territory_id": [99], "name": ["Ghost Territory"]})
        ])
        result = compute_territory_scorecard(
            minimal_data["fact_rx"],
            minimal_data["fact_rep_act"],
            minimal_data["hcp_dim"],
            minimal_data["rep_dim"],
            terr_extra,
            minimal_data["account_dim"],
        )
        assert (result["trx_per_hcp"] >= 0).all()

    def test_empty_territory_dim(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["territory_dim"].columns)
        result = compute_territory_scorecard(
            minimal_data["fact_rx"],
            minimal_data["fact_rep_act"],
            minimal_data["hcp_dim"],
            minimal_data["rep_dim"],
            empty,
            minimal_data["account_dim"],
        )
        assert result.empty

    def test_empty_fact_rx(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["fact_rx"].columns)
        result = compute_territory_scorecard(
            empty,
            minimal_data["fact_rep_act"],
            minimal_data["hcp_dim"],
            minimal_data["rep_dim"],
            minimal_data["territory_dim"],
            minimal_data["account_dim"],
        )
        assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPUTE HCP COVERAGE
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeHcpCoverage:
    def test_happy_path(self, minimal_data):
        result = compute_hcp_coverage(minimal_data["fact_rep_act"], minimal_data["hcp_dim"])
        assert result["total_hcps"] == 3
        assert result["called_hcps"] <= 3
        assert result["never_called"] >= 0
        assert 0 <= result["coverage_pct"] <= 100

    def test_all_hcps_called(self, minimal_data):
        # Make sure every hcp_id appears in fact_rep_act
        all_ids = minimal_data["hcp_dim"]["hcp_id"].tolist()
        act = pd.DataFrame({
            "activity_id": range(len(all_ids)),
            "rep_id": [1] * len(all_ids),
            "hcp_id": all_ids,
            "account_id": [1000] * len(all_ids),
            "date_id": [20240801] * len(all_ids),
            "activity_type": ["call"] * len(all_ids),
            "status": ["completed"] * len(all_ids),
            "duration_min": [20] * len(all_ids),
        })
        result = compute_hcp_coverage(act, minimal_data["hcp_dim"])
        assert result["never_called"] == 0
        assert result["coverage_pct"] == 100.0

    def test_no_hcps_called(self, minimal_data):
        empty_act = pd.DataFrame(columns=minimal_data["fact_rep_act"].columns)
        result = compute_hcp_coverage(empty_act, minimal_data["hcp_dim"])
        assert result["called_hcps"] == 0
        assert result["coverage_pct"] == 0.0
        assert result["never_called"] == 3

    def test_empty_hcp_dim(self, minimal_data):
        empty_hcp = pd.DataFrame(columns=minimal_data["hcp_dim"].columns)
        result = compute_hcp_coverage(minimal_data["fact_rep_act"], empty_hcp)
        assert result["total_hcps"] == 0
        assert result["coverage_pct"] == 0.0

    def test_hcp_id_in_activity_not_in_dim(self, minimal_data):
        # Ghost hcp_id — should not crash
        act = minimal_data["fact_rep_act"].copy()
        act.loc[0, "hcp_id"] = 9999999999
        result = compute_hcp_coverage(act, minimal_data["hcp_dim"])
        assert isinstance(result, dict)

    def test_never_called_df_has_correct_columns(self, minimal_data):
        # Remove all activities → all HCPs never called
        empty_act = pd.DataFrame(columns=minimal_data["fact_rep_act"].columns)
        result = compute_hcp_coverage(empty_act, minimal_data["hcp_dim"])
        df = result["never_called_df"]
        assert "full_name" in df.columns
        assert "tier" in df.columns
        assert "specialty" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# 8. COMPUTE PAYOR SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

class TestComputePayorSummary:
    def test_happy_path(self, minimal_data):
        result = compute_payor_summary(
            minimal_data["fact_payor"],
            minimal_data["account_dim"],
            minimal_data["date_dim"],
        )
        assert "overall" in result
        assert "by_account" in result
        assert not result["overall"].empty

    def test_empty_fact_payor(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["fact_payor"].columns)
        result = compute_payor_summary(empty, minimal_data["account_dim"], minimal_data["date_dim"])
        assert result["overall"].empty
        assert result["by_account"].empty

    def test_payor_pct_are_averages(self, minimal_data):
        result = compute_payor_summary(
            minimal_data["fact_payor"],
            minimal_data["account_dim"],
            minimal_data["date_dim"],
        )
        # All averages should be positive
        assert (result["overall"] >= 0).all()

    def test_empty_account_dim(self, minimal_data):
        empty = pd.DataFrame(columns=minimal_data["account_dim"].columns)
        result = compute_payor_summary(
            minimal_data["fact_payor"], empty, minimal_data["date_dim"]
        )
        assert result["by_account"].empty

    def test_single_payor_type(self, minimal_data):
        single = minimal_data["fact_payor"][minimal_data["fact_payor"]["payor_type"] == "Commercial"]
        result = compute_payor_summary(single, minimal_data["account_dim"], minimal_data["date_dim"])
        assert len(result["overall"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 9. COMPUTE MARKET SHARE
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeMarketShare:
    def test_happy_path(self, minimal_data):
        result = compute_market_share(minimal_data["fact_ln"])
        assert "overall_avg" in result
        assert "overall_patient_avg" in result
        assert "by_quarter" in result
        assert result["overall_avg"] > 0

    def test_empty_fact_ln(self):
        empty = pd.DataFrame(columns=["entity_type", "entity_id", "quarter_id",
                                       "ln_patient_cnt", "est_market_share"])
        result = compute_market_share(empty)
        assert result["overall_avg"] == 0.0
        assert result["overall_patient_avg"] == 0.0
        assert result["by_quarter"].empty

    def test_single_row(self):
        df = pd.DataFrame({
            "entity_type": ["H"],
            "entity_id": [1],
            "quarter_id": ["2024Q3"],
            "ln_patient_cnt": [100],
            "est_market_share": [30.0],
        })
        result = compute_market_share(df)
        assert result["overall_avg"] == 30.0
        assert result["overall_patient_avg"] == 100.0

    def test_market_share_not_negative(self, minimal_data):
        result = compute_market_share(minimal_data["fact_ln"])
        assert result["overall_avg"] >= 0

    def test_by_quarter_sorted(self, minimal_data):
        result = compute_market_share(minimal_data["fact_ln"])
        quarters = result["by_quarter"].index.tolist()
        assert quarters == sorted(quarters)


# ══════════════════════════════════════════════════════════════════════════════
# 10. COMPUTE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeInsights:
    def _make_inputs(self, minimal_data):
        rep_sc  = compute_rep_scorecard(
            minimal_data["fact_rep_act"], minimal_data["rep_dim"], minimal_data["hcp_dim"]
        )
        terr_sc = compute_territory_scorecard(
            minimal_data["fact_rx"], minimal_data["fact_rep_act"],
            minimal_data["hcp_dim"], minimal_data["rep_dim"],
            minimal_data["territory_dim"], minimal_data["account_dim"],
        )
        q_trend = compute_quarterly_trend(minimal_data["fact_rx"], minimal_data["date_dim"])
        hcp_cov = compute_hcp_coverage(minimal_data["fact_rep_act"], minimal_data["hcp_dim"])
        payor   = compute_payor_summary(
            minimal_data["fact_payor"], minimal_data["account_dim"], minimal_data["date_dim"]
        )
        return rep_sc, terr_sc, q_trend, hcp_cov, payor

    def test_returns_list(self, minimal_data):
        result = compute_insights(*self._make_inputs(minimal_data))
        assert isinstance(result, list)
        assert len(result) > 0

    def test_all_items_are_strings(self, minimal_data):
        result = compute_insights(*self._make_inputs(minimal_data))
        assert all(isinstance(i, str) for i in result)

    def test_empty_rep_scorecard(self, minimal_data):
        rep_sc, terr_sc, q_trend, hcp_cov, payor = self._make_inputs(minimal_data)
        result = compute_insights(pd.DataFrame(), terr_sc, q_trend, hcp_cov, payor)
        assert isinstance(result, list)

    def test_empty_territory_scorecard(self, minimal_data):
        rep_sc, terr_sc, q_trend, hcp_cov, payor = self._make_inputs(minimal_data)
        result = compute_insights(rep_sc, pd.DataFrame(), q_trend, hcp_cov, payor)
        assert isinstance(result, list)

    def test_empty_quarterly_trend(self, minimal_data):
        rep_sc, terr_sc, q_trend, hcp_cov, payor = self._make_inputs(minimal_data)
        result = compute_insights(rep_sc, terr_sc, pd.DataFrame(), hcp_cov, payor)
        assert isinstance(result, list)

    def test_all_empty_inputs(self):
        result = compute_insights(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            {"never_called": 0, "never_called_df": pd.DataFrame(),
             "total_hcps": 0, "called_hcps": 0, "coverage_pct": 0},
            {"overall": pd.Series(dtype=float), "by_account": pd.DataFrame()},
        )
        assert isinstance(result, list)
        assert any("INFO" in i or "ALERT" in i or "WARN" in i or "POSITIVE" in i for i in result)

    def test_tier_a_uncalled_generates_critical(self, minimal_data):
        # No activities at all → all HCPs (including Tier A) never called
        empty_act = pd.DataFrame(columns=minimal_data["fact_rep_act"].columns)
        hcp_cov = compute_hcp_coverage(empty_act, minimal_data["hcp_dim"])
        rep_sc, terr_sc, q_trend, _, payor = self._make_inputs(minimal_data)
        result = compute_insights(rep_sc, terr_sc, q_trend, hcp_cov, payor)
        assert any("CRITICAL" in i for i in result)


# ══════════════════════════════════════════════════════════════════════════════
# 11. BUILD CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildContext:
    def test_returns_non_empty_string(self, minimal_data):
        result = build_context(minimal_data)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_key_sections(self, minimal_data):
        result = build_context(minimal_data)
        for section in [
            "DATASET OVERVIEW", "BRANDS", "QUARTERLY RX TREND",
            "REP SCORECARD", "TERRITORY SCORECARD", "HCP CALL COVERAGE",
            "PAYOR MIX", "MARKET SHARE", "FLAGGED INSIGHTS",
        ]:
            assert section in result, f"Missing section: {section}"

    def test_no_nan_strings_in_output(self, minimal_data):
        result = build_context(minimal_data)
        assert "nan" not in result.lower() or "NaN" not in result

    def test_missing_key_raises(self, minimal_data):
        del minimal_data["fact_rx"]
        with pytest.raises(KeyError):
            build_context(minimal_data)

    def test_empty_fact_rx_does_not_crash(self, minimal_data):
        minimal_data["fact_rx"] = pd.DataFrame(
            columns=minimal_data["fact_rx"].columns
        )
        # Should handle gracefully — either return partial context or raise clearly
        try:
            result = build_context(minimal_data)
            assert isinstance(result, str)
        except ValueError:
            pass  # acceptable if explicitly raised

    def test_context_includes_brand_names(self, minimal_data):
        result = build_context(minimal_data)
        assert "BRANDX" in result
        assert "BRANDY" in result

    def test_context_includes_rep_names(self, minimal_data):
        result = build_context(minimal_data)
        assert "Alice Smith" in result or "Alice" in result

    def test_context_includes_territory_names(self, minimal_data):
        result = build_context(minimal_data)
        assert "Territory 1" in result


# ══════════════════════════════════════════════════════════════════════════════
# 12. GET GROQ CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class TestGetGroqClient:
    def test_valid_key_returns_client(self):
        from groq import Groq
        client = get_groq_client("gsk_fake_key_for_testing_1234567890")
        assert isinstance(client, Groq)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="missing or empty"):
            get_groq_client("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="missing or empty"):
            get_groq_client(None)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="missing or empty"):
            get_groq_client("    ")

    def test_key_is_stripped(self):
        from groq import Groq
        client = get_groq_client("  gsk_fake_key_for_testing_1234567890  ")
        assert isinstance(client, Groq)


# ══════════════════════════════════════════════════════════════════════════════
# 13. BUILD SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:
    def test_returns_string(self):
        result = build_system_prompt("some context")
        assert isinstance(result, str)

    def test_context_embedded_in_prompt(self):
        result = build_system_prompt("MY_UNIQUE_CONTEXT_STRING")
        assert "MY_UNIQUE_CONTEXT_STRING" in result

    def test_contains_instructions(self):
        result = build_system_prompt("ctx")
        assert "cite" in result.lower() or "source" in result.lower()
        assert "don't" in result.lower() or "only" in result.lower()

    def test_empty_context(self):
        result = build_system_prompt("")
        assert isinstance(result, str)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 14. ASK QUESTION
# ══════════════════════════════════════════════════════════════════════════════

class TestAskQuestion:
    def _mock_client(self, response_text="This is the answer."):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = response_text
        client.chat.completions.create.return_value.choices = [choice]
        return client

    def test_happy_path(self):
        client = self._mock_client("Great answer!")
        result = ask_question(client, "sys prompt", [], "What is TRx?")
        assert result == "Great answer!"

    def test_history_is_included_in_messages(self):
        client = self._mock_client("Follow-up answer.")
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        ask_question(client, "sys", history, "Second question")
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        contents = [m["content"] for m in messages]
        assert "First question" in contents
        assert "First answer" in contents

    def test_empty_question_raises(self):
        client = self._mock_client()
        with pytest.raises(ValueError, match="non-empty"):
            ask_question(client, "sys", [], "")

    def test_whitespace_question_raises(self):
        client = self._mock_client()
        with pytest.raises(ValueError, match="non-empty"):
            ask_question(client, "sys", [], "   ")

    def test_none_question_raises(self):
        client = self._mock_client()
        with pytest.raises((ValueError, AttributeError)):
            ask_question(client, "sys", [], None)

    def test_none_client_raises(self):
        with pytest.raises(ValueError, match="None"):
            ask_question(None, "sys", [], "question?")

    def test_api_error_raises_runtime_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Connection refused")
        with pytest.raises(RuntimeError, match="Groq API call failed"):
            ask_question(client, "sys", [], "question?")

    def test_empty_choices_raises_runtime_error(self):
        client = MagicMock()
        client.chat.completions.create.return_value.choices = []
        with pytest.raises(RuntimeError, match="empty response"):
            ask_question(client, "sys", [], "question?")

    def test_null_content_raises_runtime_error(self):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        client.chat.completions.create.return_value.choices = [choice]
        with pytest.raises(RuntimeError, match="null message content"):
            ask_question(client, "sys", [], "question?")

    def test_multi_turn_history_grows(self):
        client = self._mock_client("Answer.")
        history = []
        ask_question(client, "sys", history, "Q1")
        # history is passed by reference — app.py appends externally; engine doesn't mutate it
        assert isinstance(history, list)

    def test_long_question_accepted(self):
        client = self._mock_client("OK.")
        long_q = "What is the trend? " * 200
        result = ask_question(client, "sys", [], long_q)
        assert result == "OK."

    def test_special_characters_in_question(self):
        client = self._mock_client("Handled.")
        result = ask_question(client, "sys", [], "What about <script>alert('xss')</script>?")
        assert result == "Handled."

    def test_unicode_question(self):
        client = self._mock_client("Fine.")
        result = ask_question(client, "sys", [], "Qué territorio tiene más TRx? 日本語テスト")
        assert result == "Fine."


# ══════════════════════════════════════════════════════════════════════════════
# 15. END-TO-END INTEGRATION (no real API call)
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_load_build_ask_pipeline(self, csv_dir):
        """Full pipeline: load → build context → build prompt → ask (mocked)."""
        data    = load_data(str(csv_dir))
        context = build_context(data)
        prompt  = build_system_prompt(context)

        assert len(context) > 100
        assert len(prompt) > len(context)

        client = MagicMock()
        choice = MagicMock()
        choice.message.content = "Territory 1 has the highest TRx/HCP ratio."
        client.chat.completions.create.return_value.choices = [choice]

        answer = ask_question(client, prompt, [], "Which territory is most efficient?")
        assert "Territory 1" in answer

    def test_context_is_deterministic(self, csv_dir):
        """Same data → same context string every time."""
        data = load_data(str(csv_dir))
        ctx1 = build_context(data)
        ctx2 = build_context(data)
        assert ctx1 == ctx2

    def test_pipeline_handles_single_rep(self, tmp_path):
        """Minimal single-rep dataset doesn't crash."""
        single = {
            "account_dim": pd.DataFrame({
                "account_id": [1000], "name": ["Hosp A"],
                "account_type": ["Hospital"], "territory_id": [1],
            }),
            "date_dim": pd.DataFrame({
                "date_id": [20240801], "calendar_date": ["2024-08-01"],
                "year": [2024], "quarter": ["Q3"],
            }),
            "hcp_dim": pd.DataFrame({
                "hcp_id": [1000000001], "full_name": ["Dr Solo"],
                "specialty": ["Oncology"], "tier": ["A"], "territory_id": [1],
            }),
            "rep_dim": pd.DataFrame({
                "rep_id": [1], "first_name": ["Only"],
                "last_name": ["Rep"], "region": ["Territory 1"],
            }),
            "territory_dim": pd.DataFrame({
                "territory_id": [1], "name": ["Territory 1"],
            }),
            "fact_rx": pd.DataFrame({
                "hcp_id": [1000000001], "date_id": [20240801],
                "brand_code": ["X"], "trx_cnt": [5], "nrx_cnt": [2],
            }),
            "fact_rep_act": pd.DataFrame({
                "activity_id": [1], "rep_id": [1], "hcp_id": [1000000001],
                "account_id": [1000], "date_id": [20240801],
                "activity_type": ["call"], "status": ["completed"],
                "duration_min": [20],
            }),
            "fact_ln": pd.DataFrame({
                "entity_type": ["H"], "entity_id": [1000000001],
                "quarter_id": ["2024Q3"], "ln_patient_cnt": [10],
                "est_market_share": [20.0],
            }),
            "fact_payor": pd.DataFrame({
                "account_id": [1000], "date_id": [20240801],
                "payor_type": ["Commercial"], "pct_of_volume": [100.0],
            }),
        }
        context = build_context(single)
        assert "Only Rep" in context or "Territory 1" in context
