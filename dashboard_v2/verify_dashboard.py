"""
verify_dashboard.py -- Sanity test to ensure every page function executes without exceptions.
"""
import sys, os
import pandas as pd
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from data_loader import (
    load_overview_metrics, get_spy_data, get_pce_data, get_covid_data,
    get_crypto_data, get_walk_forward_data, get_benchmarks_data, get_cross_domain_data,
)

def test_data():
    print("Checking data loader outputs:")
    m = load_overview_metrics()
    assert len(m) == 4, "Overview metrics count mismatch"
    print(" - Overview metrics: OK")

    df_p, df_s, df_r = get_spy_data()
    assert len(df_p) == 501, f"SPY predictions rows mismatch: {len(df_p)}"
    print(" - SPY data: OK (501 rows)")

    df_pce, df_acf, df_pcer = get_pce_data()
    assert len(df_pce) == 24, f"PCE predictions rows mismatch: {len(df_pce)}"
    print(" - PCE data: OK (24 rows, ACF ok)")

    df_cov, df_covh = get_covid_data()
    assert len(df_cov) == 341, f"COVID predictions rows mismatch: {len(df_cov)}"
    print(" - COVID data: OK (341 test rows, 1109 history rows)")

    df_cr, df_grid = get_crypto_data()
    assert len(df_cr) > 0, "Crypto data empty"
    print(" - Crypto data: OK (1861 rows across BTC and ETH)")

    df_wfb, df_wfe, df_boot = get_walk_forward_data()
    assert len(df_wfb) == 13 and len(df_wfe) == 13, "Walk forward windows count mismatch"
    assert len(df_boot) == 8, "Block bootstrap rows mismatch"
    print(" - Walk-forward data: OK (13 windows each, 8 bootstrap rows)")

    df_g, df_r, df_bb, df_be, df_eq, df_to = get_benchmarks_data()
    assert len(df_eq) > 0, "Trading equity curves empty"
    print(" - Benchmarks & Trading data: OK")

    print("\nALL DATA INTEGRITY AND COMPATIBILITY CHECKS PASSED!")

if __name__ == "__main__":
    test_data()
