import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    return Interpreter, Path, csv, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Mean/std correction for raw RSS samples

    `firmware/vlp_serial/main.cpp` only ever sees **one raw reading at a time** and has
    no ground-truth label, so any preprocessing candidate has to be a pure per-sample
    function `f(raw_row) -> processed_row` — no grouping across requests, no history.

    Raw samples run at a slightly different mean and std than clean samples. This
    notebook fits and tests three ways to correct that: a single global mean shift, a
    single global mean+std match, and a per-sample (instance) mean+std normalization.
    """)
    return


@app.cell
def _(Path):
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    MODELS = ROOT / "models"

    CONF2_COLUMNS = [
        "led_0", "led_2", "led_4",
        "led_12", "led_14", "led_16",
        "led_24", "led_26", "led_28",
    ]
    return CONF2_COLUMNS, DATA, MODELS


@app.cell
def _(CONF2_COLUMNS, csv, np):
    def load_csv(path):
        rows, xy = [], []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([float(row[c]) for c in CONF2_COLUMNS])
                xy.append((float(row["x"]), float(row["y"])))
        x = np.asarray(rows, dtype=np.float32)
        y_cm = np.asarray(xy, dtype=np.float32) / 10.0
        return x, y_cm

    return (load_csv,)


@app.cell
def _(DATA, load_csv, mo, np):
    # Fit any transform only on the *train* split (never on the data we score against).
    X_train_clean, _y_train_clean = load_csv(DATA / "train_clean_3x3_1cm.csv")
    X_train_raw, _y_train_raw = load_csv(DATA / "train_raw_3x3_1cm.csv")

    # Held-out raw split used purely for evaluation.
    X_test_raw, y_test_cm = load_csv(DATA / "validation_raw_3x3_1cm.csv")

    mo.md(
        f"Loaded: clean train n={len(X_train_clean)}, raw train n={len(X_train_raw)}, "
        f"raw validation (held out) n={len(X_test_raw)}. "
        f"Clean exact-zero fraction: {np.mean(X_train_clean == 0):.3f}, "
        f"raw exact-zero fraction: {np.mean(X_train_raw == 0):.3f}."
    )
    return X_test_raw, X_train_clean, X_train_raw, y_test_cm


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-sample statistics: mean and std across channels

    For each individual row, take the mean and std **across its 9 channels**. If raw
    rows systematically run at a different "energy level" (mean) or "spread" (std) than
    clean rows, a per-sample transform like `(row - row.mean()) / row.std() * target_std
    + target_mean` (instance normalization, rescaled to clean's typical level) could
    realign a raw fingerprint's *shape* even though it never sees another sample or a
    label.
    """)
    return


@app.cell
def _(X_train_clean, X_train_raw):
    clean_sample_mean = X_train_clean.mean(axis=1)
    clean_sample_std = X_train_clean.std(axis=1)
    raw_sample_mean = X_train_raw.mean(axis=1)
    raw_sample_std = X_train_raw.std(axis=1)

    print(
        f"clean sample-mean: {clean_sample_mean.mean():.4f} +/- {clean_sample_mean.std():.4f}  "
        f"(min {clean_sample_mean.min():.4f}, max {clean_sample_mean.max():.4f})"
    )
    print(
        f"raw   sample-mean: {raw_sample_mean.mean():.4f} +/- {raw_sample_mean.std():.4f}  "
        f"(min {raw_sample_mean.min():.4f}, max {raw_sample_mean.max():.4f})"
    )
    print(
        f"clean sample-std:  {clean_sample_std.mean():.4f} +/- {clean_sample_std.std():.4f}  "
        f"(min {clean_sample_std.min():.4f}, max {clean_sample_std.max():.4f})"
    )
    print(
        f"raw   sample-std:  {raw_sample_std.mean():.4f} +/- {raw_sample_std.std():.4f}  "
        f"(min {raw_sample_std.min():.4f}, max {raw_sample_std.max():.4f})"
    )
    return clean_sample_mean, clean_sample_std, raw_sample_mean, raw_sample_std


@app.cell
def _(
    clean_sample_mean,
    clean_sample_std,
    plt,
    raw_sample_mean,
    raw_sample_std,
):
    fig_stats, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(11, 4))

    ax_mean.hist(clean_sample_mean, bins=60, alpha=0.5, density=True, label="clean")
    ax_mean.hist(raw_sample_mean, bins=60, alpha=0.5, density=True, label="raw")
    ax_mean.set_xlabel("per-sample mean (across channels)")
    ax_mean.set_ylabel("density")
    ax_mean.set_title("Per-sample mean")
    ax_mean.legend()

    ax_std.hist(clean_sample_std, bins=60, alpha=0.5, density=True, label="clean")
    ax_std.hist(raw_sample_std, bins=60, alpha=0.5, density=True, label="raw")
    ax_std.set_xlabel("per-sample std (across channels)")
    ax_std.set_title("Per-sample std")
    ax_std.legend()

    fig_stats
    return


@app.cell
def _(MODELS, np):
    scaling = np.load(MODELS / "ourmlp_task1_scaling.npz")
    rss_scale = float(scaling["rss_scale"])
    target_min_cm = scaling["target_min_cm"].astype(np.float32)
    target_range_cm = scaling["target_range_cm"].astype(np.float32)

    tflite_bytes = (MODELS / "ourmlp_task1.tflite").read_bytes()
    int8_tflite_bytes = (MODELS / "ourmlp_task1_int8.tflite").read_bytes()
    return (
        int8_tflite_bytes,
        rss_scale,
        target_min_cm,
        target_range_cm,
        tflite_bytes,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Candidate mean/std corrections

    Each transform below is a pure function of a single raw row — nothing here looks
    at neighboring samples or labels, so all of them are directly portable to the
    firmware's `handle_predict` loop.
    """)
    return


@app.cell
def _(np):
    def fit_mean_shift(raw_ref, clean_ref):
        # A single scalar added to every channel of every sample, so the raw set's
        # overall (global) mean lands on the clean set's overall mean -- no per-channel
        # or per-sample fitting, just one number.
        return float(clean_ref.mean() - raw_ref.mean())

    def apply_mean_shift(X, shift):
        return np.clip(X + shift, 0.0, None)

    def fit_global_affine(raw_ref, clean_ref):
        # Like fit_mean_shift, but also rescales by a single global std ratio --
        # one (scale, shift) pair for the whole array, not per-channel and not
        # per-sample. raw_ref.std() > clean_ref.std() here, so this compresses raw's
        # spread down to clean's; unlike instance_norm it does NOT reset each row's
        # own std -- it applies the same scale/shift everywhere.
        raw_mean, raw_std = float(raw_ref.mean()), float(raw_ref.std())
        clean_mean, clean_std = float(clean_ref.mean()), float(clean_ref.std())
        scale = clean_std / raw_std
        shift = clean_mean - raw_mean * scale
        return scale, shift

    def apply_global_affine(X, scale, shift):
        return np.clip(X * scale + shift, 0.0, None)

    def fit_instance_norm(clean_ref):
        # Target "typical fingerprint": the average per-sample mean and average
        # per-sample std seen in the clean set (not the flattened population std,
        # which would also bake in *between*-sample variation we don't want here).
        target_mean = float(clean_ref.mean(axis=1).mean())
        target_std = float(clean_ref.std(axis=1).mean())
        return target_mean, target_std

    def apply_instance_norm(X, target_mean, target_std, std_min, std_max):
        # Per-sample (instance) normalization: re-center and re-scale each raw row
        # using *its own* mean/std, then map onto clean's typical mean/std. Clamping
        # the row's std into [std_min, std_max] guards against blow-ups when a row is
        # almost all zeros (std near 0) and against over-flattening very spiky rows.
        row_mean = X.mean(axis=1, keepdims=True)
        row_std = np.clip(X.std(axis=1, keepdims=True), std_min, std_max)
        out = (X - row_mean) / row_std * target_std + target_mean
        return np.clip(out, 0.0, None)

    return (
        apply_global_affine,
        apply_instance_norm,
        apply_mean_shift,
        fit_global_affine,
        fit_instance_norm,
        fit_mean_shift,
    )


@app.cell
def _(
    X_train_clean,
    X_train_raw,
    fit_global_affine,
    fit_instance_norm,
    fit_mean_shift,
):
    mean_shift_value = fit_mean_shift(X_train_raw, X_train_clean)
    instance_norm_target = fit_instance_norm(X_train_clean)
    global_affine_params = fit_global_affine(X_train_raw, X_train_clean)
    return global_affine_params, instance_norm_target, mean_shift_value


@app.cell
def _(global_affine_params, mean_shift_value, mo):
    _scale, _shift = global_affine_params
    mo.md(f"""
    Fitted global mean shift: raw + `{mean_shift_value:.4f}` -> matches clean's overall mean

    Fitted global mean+std match: raw * `{_scale:.4f}` + `{_shift:.4f}` -> matches clean's
    overall mean and std (single scalar pair, not per-sample)
    """)
    return


@app.cell
def _(CONF2_COLUMNS, mo):
    channel_picker = mo.ui.dropdown(
        options={f"{i}: {name}": i for i, name in enumerate(CONF2_COLUMNS)},
        value="0: led_0",
        label="Channel",
    )
    channel_picker
    return (channel_picker,)


@app.cell
def _(mo):
    transform_picker = mo.ui.dropdown(
        options=["identity", "mean_shift", "global_affine", "instance_norm"],
        value="identity",
        label="Transform",
    )
    instance_norm_std_min = mo.ui.slider(0.0, 0.15, step=0.005, value=0.05, label="instance-norm std clip: min")
    instance_norm_std_max = mo.ui.slider(0.1, 0.4, step=0.005, value=0.25, label="instance-norm std clip: max")
    mo.vstack([transform_picker, instance_norm_std_min, instance_norm_std_max])
    return instance_norm_std_max, instance_norm_std_min, transform_picker


@app.cell
def _(
    X_test_raw,
    apply_global_affine,
    apply_instance_norm,
    apply_mean_shift,
    global_affine_params,
    instance_norm_std_max,
    instance_norm_std_min,
    instance_norm_target,
    mean_shift_value,
    transform_picker,
):
    def transform(X, name):
        if name == "identity":
            return X
        if name == "mean_shift":
            return apply_mean_shift(X, mean_shift_value)
        if name == "global_affine":
            scale, shift = global_affine_params
            return apply_global_affine(X, scale, shift)
        if name == "instance_norm":
            target_mean, target_std = instance_norm_target
            return apply_instance_norm(
                X, target_mean, target_std,
                instance_norm_std_min.value, instance_norm_std_max.value,
            )
        raise ValueError(name)

    X_test_transformed = transform(X_test_raw, transform_picker.value)
    return (X_test_transformed,)


@app.cell
def _(X_test_transformed, X_train_clean, channel_picker, plt):
    _c = channel_picker.value
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(X_train_clean[:, _c], bins=60, alpha=0.5, density=True, label="clean (target shape)")
    ax2.hist(X_test_transformed[:, _c], bins=60, alpha=0.5, density=True, label="transformed raw")
    ax2.set_xlabel("RSS value")
    ax2.set_ylabel("density")
    ax2.set_title(f"Channel {_c}: transformed raw vs clean target shape")
    ax2.legend()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Effect on actual position error (float + int8 exported models)
    """)
    return


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
def _(
    X_test_transformed,
    euclidean_errors_cm,
    int8_tflite_bytes,
    mo,
    np,
    rss_scale,
    run_tflite,
    target_min_cm,
    target_range_cm,
    tflite_bytes,
    y_test_cm,
):
    X_norm_for_model = np.clip(X_test_transformed, 0.0, None) / rss_scale

    float_pred_cm = run_tflite(tflite_bytes, X_norm_for_model, target_min_cm, target_range_cm, False)
    int8_pred_cm = run_tflite(int8_tflite_bytes, X_norm_for_model, target_min_cm, target_range_cm, True)

    float_errors = euclidean_errors_cm(float_pred_cm, y_test_cm)
    int8_errors = euclidean_errors_cm(int8_pred_cm, y_test_cm)

    mo.md(
        f"""
        | model | mean (cm) | median (cm) | p95 (cm) |
        |---|---|---|---|
        | float | {float_errors.mean():.3f} | {np.median(float_errors):.3f} | {np.percentile(float_errors, 95):.3f} |
        | int8  | {int8_errors.mean():.3f} | {np.median(int8_errors):.3f} | {np.percentile(int8_errors, 95):.3f} |
        """
    )
    return


if __name__ == "__main__":
    app.run()
