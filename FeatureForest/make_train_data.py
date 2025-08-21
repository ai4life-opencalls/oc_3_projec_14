import marimo

__generated_with = "0.11.31"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.callout(
        mo.vstack([
            mo.Html(
                r"""
                  <b>Note:</b>
                  <h3>You should put all your species data under a single directory and select that directory.</h3>
                """
            ).style({"font-size": "18px"}),
            mo.image(src="assets/data_dir.png", width=340)
        ])
    , kind="info")
    return


@app.cell
def _(mo):
    ui_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select the ***main data folder*** containing species folders:"
    )

    ui_browser
    return (ui_browser,)


@app.cell
def _(mo):
    get_allow_samples, set_allow_samples = mo.state(False)

    ui_level_num = mo.ui.number(start=1, stop=9, step=1, value=1, label="Downsample Level:")
    ui_num_samples = mo.ui.number(start=1, step=1, value=3, label="Number of samples:")
    ui_samples_button = mo.ui.button(
        value=get_allow_samples(),
        on_change=lambda value: set_allow_samples(True),
        label="Get Samplse...",
        kind="success",
    )

    mo.vstack([
        mo.md(
            f"- Input the downsample level for which you want to make a train dataset.<br>{mo.as_html(ui_level_num)}"
        ),
        mo.md(
            f"- Input number of samples for each species.<br>{mo.as_html(ui_num_samples)}"
        ),
        ui_samples_button
    ])
    return (
        get_allow_samples,
        set_allow_samples,
        ui_level_num,
        ui_num_samples,
        ui_samples_button,
    )


@app.cell
def _(mo, ui_browser, ui_level_num, ui_num_samples):
    # Get list of species datasets at the selected level
    raw_data_dir = ui_browser.path(0)
    level = ui_level_num.value
    num_samples = ui_num_samples.value

    species_folders = [
        f.joinpath("dataset", f"level_{level}") for f in raw_data_dir.iterdir() if f.is_dir()
    ]
    species_folders.sort()

    missing_dirs = []
    for _idx, _sp_dir in enumerate(species_folders):
        if not _sp_dir.exists():
            _dir = species_folders.pop(_idx)
            missing_dirs.append(_dir)


    _output = None
    if len(missing_dirs) > 0:
        missings = [str(m) for m in missing_dirs]
        _output = mo.vstack([
            mo.Html(
                f"⚠️ Missing datasets:<br>{mo.as_html(missings)}<br>If you don't want to include them, then ignore the warning."
            ).callout(kind="warn")
        ])

    _output
    return (
        level,
        missing_dirs,
        missings,
        num_samples,
        raw_data_dir,
        species_folders,
    )


@app.cell
def _(
    get_allow_samples,
    mo,
    num_samples,
    set_allow_samples,
    species_folders,
    take_samples,
):
    # run only on button was clicked.
    mo.stop(not get_allow_samples())

    samples = []

    with mo.redirect_stdout():
        for sp_dir in species_folders:
            print(f"Getting samples from {sp_dir}...")
            samples.extend(take_samples(sp_dir, num_samples=num_samples))

        print("Done!")

    set_allow_samples(False)
    return samples, sp_dir


@app.cell
def _(mo):
    get_allow_run, set_allow_run = mo.state(False)

    ui_save_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select a base folder to save the train samples:"
    )

    ui_save_button = mo.ui.button(
        value=get_allow_run(),
        on_change=lambda value: set_allow_run(True),
        label="Save",
        kind="success",
    )

    mo.vstack([
        mo.md(r"""Samples will be save in **`ff_train_data_level_#`** folder inside the selected base directory."""),
        ui_save_browser,
        ui_save_button
    ])
    return get_allow_run, set_allow_run, ui_save_browser, ui_save_button


@app.cell
def _(
    get_allow_run,
    level,
    mo,
    np,
    samples,
    set_allow_run,
    tifffile,
    ui_save_browser,
):
    # run only on save button was clicked.
    mo.stop(not get_allow_run())

    _output = None

    if ui_save_browser.value:
        result_dir = ui_save_browser.path(0).joinpath(f"ff_train_data_level_{level}")
        result_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(samples):
            tifffile.imwrite(result_dir.joinpath(f"sample_{i + 1}.tiff"), sample)

        stack = np.stack(samples).astype(np.uint8)
        tifffile.imwrite(
            result_dir.joinpath("train_stack.tiff"),
            stack
        )

        _output = mo.md(f"All individual samples and the stack were save in `{result_dir}`.").callout(kind="success")
    else:
        _output = mo.callout("No directory was selected!", kind="warn")

    # reset the button state/value
    set_allow_run(False)

    _output
    return i, result_dir, sample, stack


@app.cell
def _(mo):
    mo.md(r"### Utility Functions")
    return


@app.cell
def _(Path, has_content, np, pims, warnings):
    def take_samples(data_dir: Path | str, num_samples=7):
        data_dir = Path(data_dir)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            lazy_images = pims.ImageSequence(str(data_dir / "*.tiff"))
            sample_images = []

            while len(sample_images) < num_samples:
                rnd_indices = np.random.choice(len(lazy_images), size=num_samples * 10, replace=False)
                sample_images.extend(
                    [lazy_images[i] for i in rnd_indices if has_content(lazy_images[i])]
                )

        return sample_images[:num_samples]
    return (take_samples,)


@app.cell
def _(np):
    def has_content(image):
        green_ch = image[..., 1] / 255.0
        img_var = np.var(green_ch)
        num_blacks = (green_ch == 0).sum()

        return img_var > 0.012 and num_blacks < green_ch.size // 3
    return (has_content,)


@app.cell
def _():
    import warnings
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import pims
    import tifffile
    import marimo as mo


    np.random.seed(777)
    return Path, mo, np, pims, plt, tifffile, warnings


if __name__ == "__main__":
    app.run()
