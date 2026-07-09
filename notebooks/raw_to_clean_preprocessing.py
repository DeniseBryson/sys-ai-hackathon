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
    # Turning raw RSS samples into "clean-like" samples

    `firmware/vlp_serial/main.cpp` only ever sees **one raw reading at a time** and has
    no ground-truth label, so any preprocessing candidate has to be a pure per-sample
    function `f(raw_row) -> processed_row` — no grouping across requests, no history.

    Plain clipping to the training range turned out to do almost nothing: only ~1% of
    raw values in the training set actually exceed `rss_scale`, so there's barely
    anything to clip. This notebook instead compares the **marginal distribution** of
    each raw channel against the corresponding clean channel, and tries transforms that
    reshape raw samples to look more like clean ones — channel by channel, sample by
    sample.
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


app._unparsable_cell(
    r"""
    X_train_clean ´
    """,
    name="_"
)


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
    ## Per-channel distribution: clean vs. raw
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
def _(X_train_clean, X_train_raw, channel_picker, plt):
    _c = channel_picker.value
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bins = 60
    ax1.hist(X_train_clean[:, _c], bins=bins, alpha=0.5, density=True, label="clean")
    ax1.hist(X_train_raw[:, _c], bins=bins, alpha=0.5, density=True, label="raw")
    ax1.set_xlabel("RSS value")
    ax1.set_ylabel("density")
    ax1.set_title(f"Channel {_c}: clean vs raw marginal distribution")
    ax1.legend()
    fig1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Candidate per-sample transforms

    Each transform below is a pure function of a single raw row — nothing here looks
    at neighboring samples or labels, so all of them are directly portable to the
    firmware's `handle_predict` loop.
    """)
    return


@app.cell
def _(np):
    N_QUANTILES = 200
    _quantile_levels = np.linspace(0.0, 1.0, N_QUANTILES)

    def fit_quantile_maps(raw_ref, clean_ref, n_channels):
        maps = []
        for c in range(n_channels):
            raw_q = np.quantile(raw_ref[:, c], _quantile_levels)
            clean_q = np.quantile(clean_ref[:, c], _quantile_levels)
            maps.append((raw_q, clean_q))
        return maps

    def apply_quantile_maps(X, maps, blend):
        out = np.empty_like(X)
        for c, (raw_q, clean_q) in enumerate(maps):
            mapped = np.interp(X[:, c], raw_q, clean_q)
            out[:, c] = (1.0 - blend) * X[:, c] + blend * mapped
        return out

    def fit_affine(raw_ref, clean_ref, n_channels):
        params = []
        for c in range(n_channels):
            raw_mean, raw_std = raw_ref[:, c].mean(), raw_ref[:, c].std() + 1e-8
            clean_mean, clean_std = clean_ref[:, c].mean(), clean_ref[:, c].std()
            params.append((raw_mean, raw_std, clean_mean, clean_std))
        return params

    def apply_affine(X, params):
        out = np.empty_like(X)
        for c, (raw_mean, raw_std, clean_mean, clean_std) in enumerate(params):
            out[:, c] = (X[:, c] - raw_mean) / raw_std * clean_std + clean_mean
        return np.clip(out, 0.0, None)

    def apply_threshold_denoise(X, threshold):
        return np.where(X < threshold, 0.0, X)

    return (
        apply_affine,
        apply_quantile_maps,
        apply_threshold_denoise,
        fit_affine,
        fit_quantile_maps,
    )


@app.cell
def _(X_train_clean, X_train_raw, fit_affine, fit_quantile_maps):
    quantile_maps = fit_quantile_maps(X_train_raw, X_train_clean, X_train_raw.shape[1])
    affine_params = fit_affine(X_train_raw, X_train_clean, X_train_raw.shape[1])
    return affine_params, quantile_maps


@app.cell
def _(mo):
    transform_picker = mo.ui.dropdown(
        options=["identity", "clip", "threshold_denoise", "affine_match", "quantile_map"],
        value="identity",
        label="Transform",
    )
    clip_multiplier = mo.ui.slider(1.0, 2.0, step=0.05, value=1.0, label="clip upper bound (× rss_scale)")
    denoise_threshold = mo.ui.slider(0.0, 0.1, step=0.005, value=0.02, label="denoise threshold")
    quantile_blend = mo.ui.slider(0.0, 1.0, step=0.05, value=1.0, label="quantile-map blend (0=identity, 1=full map)")
    mo.vstack([transform_picker, clip_multiplier, denoise_threshold, quantile_blend])
    return clip_multiplier, denoise_threshold, quantile_blend, transform_picker


@app.cell
def _(
    X_test_raw,
    affine_params,
    apply_affine,
    apply_quantile_maps,
    apply_threshold_denoise,
    clip_multiplier,
    denoise_threshold,
    np,
    quantile_blend,
    quantile_maps,
    rss_scale,
    transform_picker,
):
    def transform(X, name):
        if name == "identity":
            return X
        if name == "clip":
            return np.clip(X, 0.0, rss_scale * clip_multiplier.value)
        if name == "threshold_denoise":
            return apply_threshold_denoise(X, denoise_threshold.value)
        if name == "affine_match":
            return apply_affine(X, affine_params)
        if name == "quantile_map":
            return apply_quantile_maps(X, quantile_maps, quantile_blend.value)
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Reference table (fixed, not tied to the sliders above)

    For context: the deployable candidates above against the un-deployable "grouped
    mean" ceiling (it needs repeated raw readings tied to a known coordinate, which the
    firmware never has at inference time — shown here only to see how much headroom
    a per-sample transform is chasing).
    """)
    return


@app.cell
def _(
    X_test_raw,
    affine_params,
    apply_affine,
    apply_quantile_maps,
    euclidean_errors_cm,
    int8_tflite_bytes,
    mo,
    np,
    quantile_maps,
    rss_scale,
    run_tflite,
    target_min_cm,
    target_range_cm,
    tflite_bytes,
    y_test_cm,
):
    def _norm(X):
        return np.clip(X, 0.0, None) / rss_scale

    def _grouped_mean_ceiling():
        keys = [tuple(row) for row in np.round(y_test_cm).astype(int)]
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        grouped_x = np.empty((len(groups), X_test_raw.shape[1]), dtype=np.float32)
        grouped_y = np.empty((len(groups), 2), dtype=np.float32)
        for g, idx in enumerate(groups.values()):
            grouped_x[g] = X_test_raw[idx].mean(axis=0)
            grouped_y[g] = y_test_cm[idx].mean(axis=0)
        return grouped_x, grouped_y

    _candidates = {
        "no preprocessing": X_test_raw,
        "quantile map (full)": apply_quantile_maps(X_test_raw, quantile_maps, 1.0),
        "affine match": apply_affine(X_test_raw, affine_params),
    }

    _rows = []
    for _name, _X in _candidates.items():
        _Xn = _norm(_X)
        _fp = run_tflite(tflite_bytes, _Xn, target_min_cm, target_range_cm, False)
        _fe = euclidean_errors_cm(_fp, y_test_cm)
        _ip = run_tflite(int8_tflite_bytes, _Xn, target_min_cm, target_range_cm, True)
        _ie = euclidean_errors_cm(_ip, y_test_cm)
        _rows.append((_name, _fe.mean(), np.median(_fe), _ie.mean(), np.median(_ie)))

    _grouped_x, _grouped_y = _grouped_mean_ceiling()
    _gxn = _norm(_grouped_x)
    _gfp = run_tflite(tflite_bytes, _gxn, target_min_cm, target_range_cm, False)
    _gfe = euclidean_errors_cm(_gfp, _grouped_y)
    _rows.append(("grouped mean (not deployable)", _gfe.mean(), np.median(_gfe), float("nan"), float("nan")))

    _table = "| candidate | float mean | float median | int8 mean | int8 median |\n|---|---|---|---|---|\n"
    for _name, _fm, _fmed, _im, _imed in _rows:
        _table += f"| {_name} | {_fm:.3f} | {_fmed:.3f} | {_im:.3f} | {_imed:.3f} |\n"

    mo.md(_table)
    return


if __name__ == "__main__":
    app.run()
