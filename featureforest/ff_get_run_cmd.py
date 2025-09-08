import marimo

__generated_with = "0.14.13"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# FeatureForest Pipeline Run Command Generator""")
    return


@app.cell
def _(mo):
    ui_data_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select the ***image folder*** containing image patches:"
    )

    mo.vstack([
        mo.md("### Select your image dataset directory:"),
        ui_data_browser
    ])
    return (ui_data_browser,)


@app.cell
def _(mo):
    ui_output_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select the output folder:"
    )

    mo.vstack([
        mo.md("### Select the output directory where the results will be stored:"),
        ui_output_browser
    ])
    return (ui_output_browser,)


@app.cell
def _(mo):
    ui_rf_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="file",
        filetypes=[".bin"],
        multiple=False,
        label="Select the ***trained RF model***:"
    )

    mo.vstack([
        mo.md("### Select the trained Random Forest model:"),
        ui_rf_browser
    ])
    return (ui_rf_browser,)


@app.cell
def _(get_available_models, mo):
    feature_models = get_available_models()

    ui_feat_model_dropdown = mo.ui.dropdown(
        label="Feature Extraction Model:",
        options=feature_models,
        value=feature_models[0],
        allow_select_none=False
    )

    ui_only_extract_chkbox = mo.ui.checkbox(label="Only Extract and Store Features (no prediction)")
    ui_no_patching_chkbox = mo.ui.checkbox(label="No Patching (treat the whole image as a single patch)", value=True)
    ui_smoothing_num = mo.ui.number(label="Smoothing iterations:", start=1, stop=999, value=25)
    ui_area_th_num = mo.ui.number(label="Area Threshold:", start=1, stop=99999, value=7)
    ui_use_sam_chkbox = mo.ui.checkbox(label="Use SAM Predictor (use SAM2 for generating final masks)", value=True)

    mo.vstack([
        ui_feat_model_dropdown,
        ui_only_extract_chkbox,
        ui_no_patching_chkbox,
        mo.md("**Post-processing:**"),
        ui_smoothing_num,
        ui_area_th_num,
        ui_use_sam_chkbox,
    ])
    return (
        ui_area_th_num,
        ui_feat_model_dropdown,
        ui_no_patching_chkbox,
        ui_only_extract_chkbox,
        ui_smoothing_num,
        ui_use_sam_chkbox,
    )


@app.cell
def _(
    mo,
    ui_area_th_num,
    ui_data_browser,
    ui_feat_model_dropdown,
    ui_no_patching_chkbox,
    ui_only_extract_chkbox,
    ui_output_browser,
    ui_rf_browser,
    ui_smoothing_num,
    ui_use_sam_chkbox,
):
    _output = None

    if not ui_data_browser.value:
        _output = mo.callout("No input data directory was selected!", kind="danger")

    if not ui_output_browser.value:
        _output = mo.callout("No output directory was selected!", kind="danger")

    if not ui_rf_browser.value:
        _output = mo.callout("No RF model was selected!", kind="danger")

    cmd = "python run_pipeline.py"
    cmd += f' --data="{ui_data_browser.path(0)}"'
    cmd += f' --outdir="{ui_output_browser.path(0)}"'
    cmd += f" --feat_model={ui_feat_model_dropdown.value}"

    if ui_no_patching_chkbox.value:
        cmd += f" --no_patching"

    if ui_only_extract_chkbox.value:
        cmd += " --only_extract"
    else:
        cmd += f' --rf_model="{ui_rf_browser.path(0)}"'
        cmd += f" --smoothing_iterations={ui_smoothing_num.value}"
        cmd += f" --area_threshold={ui_area_th_num.value}"
        if ui_use_sam_chkbox.value:
            cmd += f" --post_sam"

    _output = mo.vstack([
        mo.md("**Pipeline Run Command:**"),
        mo.md(f"`{cmd}`").callout(kind="success")
    ])

    _output
    return


@app.cell
def _():
    import marimo as mo

    from featureforest.models import get_available_models
    return get_available_models, mo


if __name__ == "__main__":
    app.run()
