import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    return Interpreter, Path, csv, mo, np, plt, sys


@app.cell
def _(Path, sys):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from vlp_hackathon.aging import AgingConfig, age_rss_episodes

    return ROOT, AgingConfig, age_rss_episodes


@app.cell
def _(mo):
    mo.md(r"""
    # Counteracting LED aging drift

    `vlp_hackathon/aging.py` models each LED channel dimming independently over its
    deployment life: `reading = original * exp(-hours * k_j)`, where `k_j` comes from a
    per-channel L90 lifetime (`T90` in [10,000h, 50,000h]) that's fixed for an
    installation but drawn randomly per LED. Task 4 (`run_submission.py --task 4`)
    streams the model through contiguous episodes of increasing installation age and
    scores how badly the frozen model degrades.

    Unlike the raw/clean sensor-noise problem, this isn't fixable with a single
    calibrated constant: the decay is per-channel and time-varying, and the firmware
    never knows the true installation age or any channel's `T90`. But it *does* see a
    long stream of requests over its deployed life, so this is the first place in this
    project where a **stateful** correction — not a stateless per-sample function — makes
    sense: track a running estimate of each channel's current peak brightness and rescale
    incoming readings to compensate for however much it has sagged below the fresh-install
    reference.
    """)
    return


@app.cell
def _(Path, ROOT, csv, np):
    DATA = ROOT / "data"
    MODELS = ROOT / "models"

    CONF2_COLUMNS = [
        "led_0", "led_2", "led_4",
        "led_12", "led_14", "led_16",
        "led_24", "led_26", "led_28",
    ]

    def load_csv(path: Path):
        rows, xy = [], []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([float(row[c]) for c in CONF2_COLUMNS])
                xy.append((float(row["x"]), float(row["y"])))
        x = np.asarray(rows, dtype=np.float32)
        y_cm = np.asarray(xy, dtype=np.float32) / 10.0
        return x, y_cm

    # Task 4 starts from the *clean* 3x3 dataset, then ages it -- no raw sensor
    # noise is involved here, just the multiplicative decay (+ small flicker/noise
    # aging.py adds on top).
    X_train_clean, _y_train_clean = load_csv(DATA / "train_clean_3x3_1cm.csv")
    X_val_clean, y_val_cm = load_csv(DATA / "validation_clean_3x3_1cm.csv")

    scaling = np.load(MODELS / "ourmlp_task1_scaling.npz")
    rss_scale = float(scaling["rss_scale"])
    target_min_cm = scaling["target_min_cm"].astype(np.float32)
    target_range_cm = scaling["target_range_cm"].astype(np.float32)
    tflite_bytes = (MODELS / "ourmlp_task1.tflite").read_bytes()
    int8_tflite_bytes = (MODELS / "ourmlp_task1_int8.tflite").read_bytes()

    return (
        MODELS,
        X_train_clean,
        X_val_clean,
        int8_tflite_bytes,
        rss_scale,
        target_min_cm,
        target_range_cm,
        tflite_bytes,
        y_val_cm,
    )


@app.cell
def _(mo):
    episode_count_slider = mo.ui.slider(3, 20, step=1, value=10, label="aging episodes (matches --aging-episodes)")
    episode_count_slider
    return (episode_count_slider,)


@app.cell
def _(AgingConfig, X_val_clean, age_rss_episodes, episode_count_slider):
    # Same defaults run_submission.py uses for Task 4: max_hours=50_000, seed=123.
    aging = age_rss_episodes(
        X_val_clean,
        config=AgingConfig(max_hours=50_000.0, seed=123),
        episode_count=episode_count_slider.value,
    )
    return (aging,)


@app.cell
def _(Interpreter, np):
    def euclidean_errors_cm(pred_cm, y_cm):
        return np.sqrt(np.sum((pred_cm - y_cm) ** 2, axis=1))

    def run_tflite(model_bytes, X_norm, target_min_cm, target_range_cm, quantized):
        interp = Interpreter(model_content=model_bytes)
        interp.allocate_tensors()
        in_info = interp.get_input_details()[0]
        out_info = interp.get_output_details()[0]
        pred_norm = np.empty((len(X_norm), 2), dtype=np.float32)
        for i, row in enumerate(X_norm):
            if quantized:
                scale, zero_point = in_info["quantization"]
                q = np.round(row / scale + zero_point)
                q = np.clip(q, -128, 127).astype(np.int8)
                interp.set_tensor(in_info["index"], q[None, :])
            else:
                interp.set_tensor(in_info["index"], row[None, :].astype(np.float32))
            interp.invoke()
            raw_out = interp.get_tensor(out_info["index"])[0]
            if quantized:
                scale, zero_point = out_info["quantization"]
                raw_out = (raw_out.astype(np.float32) - zero_point) * scale
            pred_norm[i] = raw_out
        return target_min_cm + pred_norm * target_range_cm

    return euclidean_errors_cm, run_tflite


@app.cell
def _(mo):
    mo.md(r"""
    ## Baseline: uncorrected error vs. installation age

    Each episode is one fixed decay profile applied to every sample in it, so within an
    episode the true per-channel ceiling is constant -- only flicker and sensor noise
    vary sample to sample.
    """)
    return


@app.cell
def _(
    aging,
    euclidean_errors_cm,
    np,
    rss_scale,
    run_tflite,
    target_min_cm,
    target_range_cm,
    tflite_bytes,
    y_val_cm,
):
    X_uncorrected_norm = np.clip(aging.x, 0.0, None) / rss_scale
    pred_uncorrected_cm = run_tflite(tflite_bytes, X_uncorrected_norm, target_min_cm, target_range_cm, False)
    errors_uncorrected = euclidean_errors_cm(pred_uncorrected_cm, y_val_cm)
    return (errors_uncorrected,)


@app.cell
def _(np):
    def fit_reference_peak(clean_ref, percentile):
        # The "fresh install" ceiling per channel: a high percentile (not the max,
        # to be a little robust to any single outlier reading) of the clean training
        # distribution, which represents un-aged near-LED readings.
        return np.percentile(clean_ref, percentile, axis=0).astype(np.float32)

    def peak_track_correct(X, reference_peak, release, max_factor):
        # Peak-hold-with-slow-release (classic AGC envelope follower): the estimate
        # snaps up instantly to any reading that exceeds it (we just confirmed the
        # true current ceiling), and otherwise leaks down by `release` every step so
        # a permanently dimmer LED is eventually re-discovered instead of being stuck
        # at a stale high estimate forever. `release` close to 1.0 = slow forgetting.
        n = X.shape[0]
        peak_est = reference_peak.copy()
        corrected = np.empty_like(X)
        factors = np.empty_like(X)
        for i in range(n):
            row = X[i]
            peak_est = np.maximum(row, peak_est * release)
            factor = np.clip(reference_peak / np.maximum(peak_est, 1e-6), 1.0, max_factor)
            corrected[i] = row * factor
            factors[i] = factor
        return corrected, factors

    return fit_reference_peak, peak_track_correct


@app.cell
def _(mo):
    mo.md(r"""
    ## Peak-tracking correction

    Tune the release rate (how slowly the peak estimate forgets a high reading) and
    the correction cap (guards against runaway amplification if the estimate briefly
    collapses). Higher release (closer to 1) reacts more slowly but is less noisy;
    lower release reacts faster but chases position noise instead of real decay.
    """)
    return


@app.cell
def _(mo):
    release_slider = mo.ui.slider(0.990, 0.9999, step=0.0001, value=0.999, label="release rate")
    max_factor_slider = mo.ui.slider(1.0, 5.0, step=0.1, value=3.0, label="max correction factor")
    reference_percentile_slider = mo.ui.slider(90.0, 100.0, step=0.5, value=99.0, label="reference peak percentile")
    mo.vstack([release_slider, max_factor_slider, reference_percentile_slider])
    return max_factor_slider, reference_percentile_slider, release_slider


@app.cell
def _(
    X_train_clean,
    fit_reference_peak,
    reference_percentile_slider,
):
    reference_peak = fit_reference_peak(X_train_clean, reference_percentile_slider.value)
    return (reference_peak,)


@app.cell
def _(
    aging,
    max_factor_slider,
    peak_track_correct,
    reference_peak,
    release_slider,
):
    corrected_x, correction_factors = peak_track_correct(
        aging.x, reference_peak, release_slider.value, max_factor_slider.value,
    )
    return (corrected_x,)


@app.cell
def _(
    corrected_x,
    euclidean_errors_cm,
    np,
    rss_scale,
    run_tflite,
    target_min_cm,
    target_range_cm,
    tflite_bytes,
    y_val_cm,
):
    X_corrected_norm = np.clip(corrected_x, 0.0, None) / rss_scale
    pred_corrected_cm = run_tflite(tflite_bytes, X_corrected_norm, target_min_cm, target_range_cm, False)
    errors_corrected = euclidean_errors_cm(pred_corrected_cm, y_val_cm)
    return (errors_corrected,)


@app.cell
def _(aging, errors_corrected, errors_uncorrected, np, plt):
    _hours = aging.episode_hours
    _uncorrected_mean = np.array([
        errors_uncorrected[aging.episode_ids == eid].mean() for eid in range(len(_hours))
    ])
    _corrected_mean = np.array([
        errors_corrected[aging.episode_ids == eid].mean() for eid in range(len(_hours))
    ])

    fig_aging, ax_aging = plt.subplots(figsize=(7, 4))
    ax_aging.plot(_hours, _uncorrected_mean, marker="o", label="uncorrected")
    ax_aging.plot(_hours, _corrected_mean, marker="o", label="peak-tracking corrected")
    ax_aging.set_xlabel("installation age (hours)")
    ax_aging.set_ylabel("mean position error (cm)")
    ax_aging.set_title("Mean error vs. installation age")
    ax_aging.legend()
    fig_aging
    return


@app.cell
def _(aging, errors_corrected, errors_uncorrected, mo, np):
    _rows = []
    for _eid, _hrs in enumerate(aging.episode_hours):
        _mask = aging.episode_ids == _eid
        _rows.append((
            _hrs,
            errors_uncorrected[_mask].mean(),
            errors_corrected[_mask].mean(),
        ))

    _table = "| hours | uncorrected mean (cm) | corrected mean (cm) |\n|---|---|---|\n"
    for _hrs, _u, _c in _rows:
        _table += f"| {_hrs:.0f} | {_u:.3f} | {_c:.3f} |\n"

    mo.md(_table)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sanity check: cost on fresh (unaged) data

    The tracker can't distinguish "far from this LED right now" from "this LED has
    dimmed" until enough samples confirm the true ceiling, so it isn't free even when
    there's no aging at all. This shows that one-time cost against the no-aging
    baseline.
    """)
    return


@app.cell
def _(
    X_train_clean,
    X_val_clean,
    euclidean_errors_cm,
    max_factor_slider,
    np,
    peak_track_correct,
    reference_peak,
    release_slider,
    rss_scale,
    run_tflite,
    target_min_cm,
    target_range_cm,
    tflite_bytes,
    y_val_cm,
):
    X_fresh_norm = np.clip(X_val_clean, 0.0, None) / rss_scale
    fresh_pred_uncorrected = run_tflite(tflite_bytes, X_fresh_norm, target_min_cm, target_range_cm, False)
    fresh_errors_uncorrected = euclidean_errors_cm(fresh_pred_uncorrected, y_val_cm)

    fresh_corrected_x, _ = peak_track_correct(
        X_val_clean, reference_peak, release_slider.value, max_factor_slider.value,
    )
    X_fresh_corrected_norm = np.clip(fresh_corrected_x, 0.0, None) / rss_scale
    fresh_pred_corrected = run_tflite(tflite_bytes, X_fresh_corrected_norm, target_min_cm, target_range_cm, False)
    fresh_errors_corrected = euclidean_errors_cm(fresh_pred_corrected, y_val_cm)

    print(f"fresh, uncorrected:  mean={fresh_errors_uncorrected.mean():.3f}cm  median={np.median(fresh_errors_uncorrected):.3f}cm")
    print(f"fresh, corrected:    mean={fresh_errors_corrected.mean():.3f}cm  median={np.median(fresh_errors_corrected):.3f}cm")
    return


if __name__ == "__main__":
    app.run()
