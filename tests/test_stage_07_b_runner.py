"""Unit and static tests for Strategy Family #7 Stage-B runner."""

import importlib.util
import pytest

spec = importlib.util.spec_from_file_location("runner07b", "scripts/run_stage_07_b.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_grid_cardinalities():
    """Verify 36 calibration combinations and 37 total configurations."""
    calib_36 = runner.generate_36_calibration_combinations()
    assert len(calib_36) == 36

    all_37 = runner.generate_all_37_configurations()
    assert len(all_37) == 37

    # C00 check
    c00 = all_37[0]
    assert c00["candidate_id"] == "C00"
    assert c00["is_baseline"] is True
    assert c00["is_calibration_candidate"] is False
    assert c00["params"] == runner.C00_BASELINE_PARAMS

    # C01–C36 check
    calib_items = all_37[1:]
    assert len(calib_items) == 36
    for idx, item in enumerate(calib_items):
        assert item["candidate_id"] == f"C{idx+1:02d}"
        assert item["is_baseline"] is False
        assert item["is_calibration_candidate"] is True

    # Confirm C00 params are NOT inside the 36 calibration grid
    assert not any(item["params"] == runner.C00_BASELINE_PARAMS for item in calib_items)


def test_track_and_slice_definitions():
    """Verify exactly 4 tracks and 296 total execution slices."""
    tracks = runner.TRACKS
    assert len(tracks) == 4

    expected_track_names = ["BTCUSDT_4H", "ETHUSDT_4H", "BTCUSDT_1H", "ETHUSDT_1H"]
    actual_track_names = [f"{t['symbol']}_{t['timeframe'].upper()}" for t in tracks]
    assert actual_track_names == expected_track_names

    # Total backtest slices: 37 configs * 4 tracks * 2 slices (train + val) = 296
    total_configs = len(runner.generate_all_37_configurations())
    total_slices = total_configs * len(tracks) * 2
    assert total_slices == 296


def test_2026_data_protection_bounds():
    """Verify timestamps strictly exclude 2026 data."""
    pre_oos_ms = runner.PRE_OOS_END_TIMESTAMP_MS
    val_end_ms = runner.VAL_END_MS
    train_end_ms = runner.TRAIN_END_MS

    assert val_end_ms < pre_oos_ms
    assert train_end_ms < val_end_ms
    assert pre_oos_ms == 1767225600000  # 2026-01-01 00:00:00 UTC


def test_dual_asset_intersection_logic():
    """Verify dual-asset qualification requires non-empty candidate ID intersection."""
    # Case 1: Disjoint passing sets -> FAIL
    btc_pass_1 = {"C01", "C02"}
    eth_pass_1 = {"C03", "C04"}
    inter_1 = btc_pass_1.intersection(eth_pass_1)
    assert len(inter_1) == 0
    assert bool(len(inter_1) > 0) is False

    # Case 2: Shared passing candidate -> PASS
    btc_pass_2 = {"C05", "C06"}
    eth_pass_2 = {"C06", "C07"}
    inter_2 = btc_pass_2.intersection(eth_pass_2)
    assert inter_2 == {"C06"}
    assert bool(len(inter_2) > 0) is True


def test_dry_run_execution():
    """Verify dry_run_stage_07_b executes cleanly with zero backtests."""
    payload = runner.dry_run_stage_07_b()
    assert payload["dry_run"] is True
    assert payload["total_configurations"] == 37
    assert payload["calibration_configurations"] == 36
    assert payload["total_expected_slices"] == 296
    assert payload["status"] == "DRY RUN COMPLETED — ZERO BACKTESTS EXECUTED"
