import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _(mo):
    # input folders

    input_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=True,
        label="Select all folders containing tiles within region for all species"
    )

    mo.vstack([
        mo.md("### Select all folders containing filtered tiles of each species"),
        mo.md("`.../species_1/tiles_within_region`<br>`.../species_2/tiles_within_region`<br>`...`"),
        input_browser
    ])
    return (input_browser,)


@app.cell
def _(input_browser):
    data_folders = []

    for item in input_browser.value:
        data_folders.append(item.path)

    # make sure all folders are exist
    assert all([d.exists() for d in data_folders])
    return (data_folders,)


@app.cell
def _(data_folders, mo, np):
    # get all tile images and label masks
    all_images = []
    all_labels = []

    for dir in data_folders:
        all_images.extend(
            list(dir.joinpath("images").glob("*.tif"))
        )
        all_labels.extend(
            list(dir.joinpath("labels").glob("*.tif"))
        )
        assert len(all_images) == len(all_labels)

    # split dataset into tarin and test
    n_total = len(all_images)
    test_ratio = 0.1
    n_test = int(n_total * test_ratio)
    # randomly select train and test data
    indices = np.random.permutation(n_total)
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]

    assert not np.isin(train_indices, test_indices).any()

    mo.md(
        f"`Total number of tile images: {len(all_images)}`<br>"
        f"`Number of train images: {len(train_indices)}`<br>"
        f"`Number of test images: {len(test_indices)}`"
    )
    return all_images, all_labels, test_indices, train_indices


@app.cell
def _(mo):
    # where to create the datasets

    get_allow_run, set_allow_run = mo.state(False)

    output_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select a folder to create the datasets there..."
    )

    run_button = mo.ui.button(
        value=get_allow_run(),
        on_change=lambda value: set_allow_run(True),
        label="Save Datasets",
        kind="success",
    )

    mo.vstack([
        mo.md("### Select the destination folder to save the datasets there"),
        output_browser,
        run_button
    ])
    return get_allow_run, output_browser, set_allow_run


@app.cell
def _(
    all_images,
    all_labels,
    get_allow_run,
    mo,
    output_browser,
    set_allow_run,
    shutil,
    test_indices,
    train_indices,
):
    # run only on save button clicked and output dir is selected!
    mo.stop(not get_allow_run() or not output_browser.value)

    # disarm the run button
    set_allow_run(False)

    # create dataset folders
    base_dir = output_browser.path(0).joinpath("vessel_dataset")
    # train
    train_image_dir = base_dir.joinpath("train", "images")
    train_image_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir = base_dir.joinpath("train", "labels")
    train_label_dir.mkdir(parents=True, exist_ok=True)
    # test
    test_image_dir = base_dir.joinpath("test", "images")
    test_image_dir.mkdir(parents=True, exist_ok=True)
    test_label_dir = base_dir.joinpath("test", "labels")
    test_label_dir.mkdir(parents=True, exist_ok=True)

    # copy images & labels to the selected path
    with mo.redirect_stdout():
        mo.output.append(" ")
        # copy train data
        for train_count, train_idx in enumerate(train_indices):
            mo.output.replace_at_index(
                f"copying train images: {train_count + 1} / {len(train_indices)}", -1
            )
            shutil.copyfile(
                src=all_images[train_idx],
                dst=train_image_dir.joinpath(f"{train_count:04}.tif")
            )
            shutil.copyfile(
                src=all_labels[train_idx],
                dst=train_label_dir.joinpath(f"{train_count:04}.tif")
            )
        mo.output.append(" ")

        # copy test data
        for test_count, test_idx in enumerate(test_indices):
            mo.output.replace_at_index(
                f"copying test images: {test_count + 1} / {len(test_indices)}", -1
            )
            shutil.copyfile(
                src=all_images[test_idx],
                dst=test_image_dir.joinpath(f"{test_count:04}.tif")
            )
            shutil.copyfile(
                src=all_labels[test_idx],
                dst=test_label_dir.joinpath(f"{test_count:04}.tif")
            )
        mo.output.append(mo.md("<br>"))

    mo.output.append(
        mo.md(
            f"#### Your `train` and `test` datasets are created in `{base_dir}`"
        )
    )
    return


@app.cell
def _():
    import shutil
    from pathlib import Path

    import marimo as mo
    import numpy as np


    np.random.seed(777)
    return mo, np, shutil


if __name__ == "__main__":
    app.run()
