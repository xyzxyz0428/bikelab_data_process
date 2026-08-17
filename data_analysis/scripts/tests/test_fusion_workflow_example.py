"""Unit tests for the fusion workflow metric helpers."""

import importlib.util
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "fusion_workflow_example.py"
SPEC = importlib.util.spec_from_file_location("fusion_workflow_example", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frame_statistics_uses_displayed_header_interval():
    frame = pd.DataFrame({"t_ns": [10, 500_000_010, 1_000_000_010]})
    count, rate = MODULE.frame_statistics(frame)
    assert count == 3
    assert math.isclose(rate, 2.0)


def test_direct_turn_yaw_cog_metrics_wraps_at_pi():
    course = pd.DataFrame({
        "t_ns": [0, 1_000_000_000, 2_000_000_000],
        "yaw_rad": np.deg2rad([179.0, -179.0, -177.0]),
    })
    frames = {
        "Group 2": pd.DataFrame({
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "yaw_rad": np.deg2rad([-179.0, -177.0, -175.0]),
        }),
        "Group 3": pd.DataFrame({
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "yaw_rad": np.deg2rad([178.0, -180.0, -178.0]),
        }),
    }
    result = MODULE.direct_turn_yaw_cog_metrics(frames, course).set_index(
        "group"
    )
    assert math.isclose(
        result.loc["Group 2", "median_abs_yaw_cog_difference_deg"],
        2.0,
        abs_tol=1e-10,
    )
    assert math.isclose(
        result.loc["Group 3", "median_abs_yaw_cog_difference_deg"],
        1.0,
        abs_tol=1e-10,
    )


def test_bool_parameter_rejects_non_boolean_value():
    try:
        MODULE.bool_parameter({"enabled": "yes"}, "enabled")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")


def test_effective_config_checks_embedded_content_hash(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("value: 1\n", encoding="utf-8")
    content = "value: 2\n"
    encoded = content.encode("utf-8")
    provenance = {
        "effective_configs": [{
            "path": str(config),
            "exists": True,
            "type": "file",
            "content": content,
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }]
    }
    selected = MODULE.effective_config(provenance, config)
    assert selected["content"] == content


def test_read_repeatability_requires_interval_json(tmp_path):
    path = tmp_path / "repeatability.json"
    path.write_text(json.dumps({
        "start_ns": 1,
        "end_ns": 2,
        "repeatability": [{"variant": "raw_gyro_z", "repeat_count": 3}],
    }), encoding="utf-8")
    row, interval = MODULE.read_repeatability_summary(path)
    assert row["repeat_count"] == 3
    assert interval == {"start_ns": 1, "end_ns": 2}


def test_yaw_update_diagnostic_links_negative_step_to_course_update():
    step_ns = 50_000_000
    frames = {
        "Group 3": pd.DataFrame({
            "t_ns": np.arange(5, dtype=np.int64) * step_ns,
            "yaw_rad": np.deg2rad([0.0, 1.0, 2.0, 0.0, 1.0]),
        })
    }
    course = pd.DataFrame({
        "t_ns": [3 * step_ns, 4 * step_ns],
        "record_ns": [3 * step_ns + 10_000_000, 4 * step_ns + 10_000_000],
        "yaw_rad": np.deg2rad([0.0, 1.0]),
    })
    raw_gyro = pd.DataFrame({
        "t_ns": np.arange(5, dtype=np.int64) * step_ns,
        "record_ns": np.arange(5, dtype=np.int64) * step_ns + 1_000_000,
        "yaw_rate_rad_s": np.deg2rad(np.full(5, 20.0)),
    })

    result = MODULE.yaw_update_diagnostic(frames, course, raw_gyro)

    assert result["group3_negative_yaw_step_count"] == 1
    assert result["negative_steps_near_course_count"] == 1
    assert math.isclose(result["negative_steps_near_course_fraction"], 1.0)


def test_pairwise_position_separation_uses_overlapping_timestamps():
    frames = {
        "Group 2": pd.DataFrame({
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
        }),
        "Group 3": pd.DataFrame({
            "t_ns": [0, 2_000_000_000],
            "x_m": [0.0, 2.0],
            "y_m": [1.0, 1.0],
        }),
    }
    result = MODULE.pairwise_position_separation(frames).iloc[0]
    assert result["sample_count"] == 2
    assert math.isclose(result["maximum_separation_m"], 1.0)
