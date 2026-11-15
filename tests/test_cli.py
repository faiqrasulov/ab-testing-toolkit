import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.cli import main
from abkit.power import sample_size_two_proportions

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, n_a: int, n_b: int, seed: int = 0, groups=("control", "treatment")):
    rng = np.random.default_rng(seed)
    pre = rng.gamma(2.0, 10.0, n_a + n_b)
    df = pd.DataFrame(
        {
            "variant": [groups[0]] * n_a + [groups[1]] * n_b,
            "converted": rng.binomial(1, 0.1, n_a + n_b),
            "revenue": pre * 0.8 + rng.gamma(1.0, 5.0, n_a + n_b),
            "pre_revenue": pre,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_analyze_binary_report(tmp_path, capsys):
    csv = _write_csv(tmp_path / "exp.csv", 2000, 2000)
    out_md = tmp_path / "report.md"
    code = main(["analyze", str(csv), "--group", "variant", "--metric", "converted",
                 "--output", str(out_md)])
    text = capsys.readouterr().out
    assert code == 0
    for heading in ("Sample ratio mismatch", "Result (fixed-horizon test)", "Bayesian view",
                    "Power check", "Decision guidance", "P(B > A)"):
        assert heading in text
    assert "metric type: binary" in text
    assert text == out_md.read_text() + "\n"


def test_analyze_continuous_with_cuped(tmp_path, capsys):
    csv = _write_csv(tmp_path / "exp.csv", 1500, 1500)
    main(["analyze", str(csv), "--group", "variant", "--metric", "revenue",
          "--pre-metric", "pre_revenue"])
    text = capsys.readouterr().out
    assert "CUPED adjustment" in text and "metric type: continuous" in text


def test_analyze_flags_srm(tmp_path, capsys):
    csv = _write_csv(tmp_path / "exp.csv", 2300, 1700)
    main(["analyze", str(csv), "--group", "variant", "--metric", "converted"])
    text = capsys.readouterr().out
    assert "SAMPLE RATIO MISMATCH" in text and "Do not make a decision" in text


def test_analyze_rejects_more_than_two_groups(tmp_path, capsys):
    csv = _write_csv(tmp_path / "exp.csv", 100, 100)
    df = pd.read_csv(csv)
    df.loc[:9, "variant"] = "other"
    df.to_csv(csv, index=False)
    assert main(["analyze", str(csv), "--group", "variant", "--metric", "converted"]) == 2
    assert "exactly 2 groups" in capsys.readouterr().err


def test_sample_size_command(capsys):
    main(["sample-size", "--baseline", "0.10", "--relative-mde", "0.05"])
    n = sample_size_two_proportions(0.10, 0.105)
    assert f"control n = {n:,}" in capsys.readouterr().out


def test_example_generator_and_shipped_example(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("gen", ROOT / "examples" / "make_example_data.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.main(["--out", str(tmp_path)])
    fresh = pd.read_csv(tmp_path / "simulated_ab_test.csv")
    shipped = pd.read_csv(ROOT / "examples" / "simulated_ab_test.csv")
    pd.testing.assert_frame_equal(fresh, shipped)  # committed example is reproducible
    main(["analyze", str(ROOT / "examples" / "simulated_ab_test_srm.csv"), "--group", "variant",
          "--metric", "converted"])
    assert "SAMPLE RATIO MISMATCH" in capsys.readouterr().out


@pytest.mark.parametrize("control", ["treatment", None])
def test_control_selection(tmp_path, capsys, control):
    csv = _write_csv(tmp_path / "exp.csv", 500, 500)
    args = ["analyze", str(csv), "--group", "variant", "--metric", "converted"]
    main(args + (["--control", control] if control else []))
    expected = control or "control"
    assert f"control (A) = `{expected}`" in capsys.readouterr().out
