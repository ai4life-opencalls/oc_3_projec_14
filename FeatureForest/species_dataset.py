import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select the ***species folder*** (contains anatomy & images):",
    )

    browser
    return (browser,)


@app.cell
def _(mo):
    get_allow_run, set_allow_run = mo.state(False)

    ui_level_num = mo.ui.number(
        start=1, stop=9, step=1, value=1, label="Downsampling Level:"
    )

    run_button = mo.ui.button(
        value=get_allow_run(),
        on_change=lambda value: set_allow_run(True),
        label="Extract Patches",
        kind="success",
    )

    mo.vstack(
        [
            mo.md(
                "**Set the downsample level (1 means no downsampling which is the highest resolution)**"
            ),
            mo.md(
                "If you're making a dataset for the *Ray* cells, it's recommended to use higher downsampling level like 4."
            ),
            ui_level_num,
            run_button,
        ]
    )
    return get_allow_run, set_allow_run, ui_level_num


@app.cell
def _(browser, get_allow_run, get_samples, mo, set_allow_run, ui_level_num):
    mo.stop(not get_allow_run())

    species_dir = browser.path(0)
    mirax_file = list(species_dir.glob("*/*.mrxs"))[0]

    assert mirax_file.exists(), "couldn't find the mirax file!"

    level = ui_level_num.value

    dataset_dir = species_dir.joinpath("dataset", f"level_{level}")
    dataset_dir.mkdir(exist_ok=True, parents=True)

    patch_size = 512

    get_samples(mirax_file, dataset_dir, patch_size, level=level)

    # reset the button state/value
    set_allow_run(False)
    return


@app.cell
def _(mo):
    mo.md(r"""### Utility Functions""")
    return


@app.cell
def _(get_patch_coords, get_region, mo, openslide, tifffile):
    def get_samples(slide_file, save_dir, patch_size=512, level=1):
        slide = openslide.OpenSlide(slide_file)
        # actual levels are 0-index-based, between 0-8
        level_idx = level - 1
        with mo.redirect_stdout():
            patch_xs, patch_ys = get_patch_coords(slide_file, patch_size, level_idx)
        var_threshold = 0.02
        if level_idx > 0:
            var_threshold = 0.005
        # extracting patches from the slide
        with mo.redirect_stdout():
            print("\nexctracting patches:")
            for row, py in enumerate(patch_ys):
                for col, px in enumerate(patch_xs):
                    mo.output.replace_at_index(f"\trow: {row + 1:03}/{len(patch_ys)} , col: {col + 1:03}/{len(patch_xs)}", -1)

                    img = get_region(
                        slide,
                        (px, py),  # (x, y) coordinates
                        (patch_size, patch_size),  # (width, height)
                        level=level_idx,  # resolution level index
                    )
                    # if has_content(img, var_threshold):
                    tifffile.imwrite(
                        save_dir.joinpath(f"row_{row:03}_col_{col:03}.tiff"),
                        img[..., :-1],  # omit the alpha channel
                    )

    return (get_samples,)


@app.cell
def _(np, openslide):
    def get_patch_coords(slide_file, patch_size=512, level=0):
        base_name = slide_file.parent.parent.name.replace(" ", "_")
        slide = openslide.OpenSlide(slide_file)
        # get boundary of the whole image in the slide
        bound_x1 = int(slide.properties["openslide.bounds-x"])
        bound_y1 = int(slide.properties["openslide.bounds-y"])
        bound_x2 = bound_x1 + int(slide.properties["openslide.bounds-width"])
        bound_y2 = bound_y1 + int(slide.properties["openslide.bounds-height"])
        w = bound_x2 - bound_x1
        h = bound_y2 - bound_y1
        print(f"\n{base_name} / {slide_file.name}\nwidth: {w}\nheight: {h}")
        print(f"slide image boundary: {bound_x1}, {bound_y1} to {bound_x2}, {bound_y2}")

        # extract patches only for down-left quarter of the slide
        sampling_x1 = bound_x1  # 0
        sampling_y1 = bound_y1 + (h // 2)  # h // 2
        sampling_x2 = sampling_x1 + (w // 2)  # w // 2
        sampling_y2 = sampling_y1 + (h // 2)  # h
        print(
            f"sampling boundary: {sampling_x1}, {sampling_y1} to {sampling_x2}, {sampling_y2}"
        )

        # get patch size based on resolution level
        patch_size = patch_size * (2**level)
        print(f"patch size at level {level}: {patch_size}")

        # calculate padding for the region to be divisible by `patch_size`
        pad_w = patch_size - ((sampling_x2 - sampling_x1) % patch_size)
        pad_h = patch_size - ((sampling_y2 - sampling_y1) % patch_size)

        # get patches top-left coords
        patch_xs = np.arange(sampling_x1, sampling_x2 + pad_w, patch_size)
        patch_ys = np.arange(sampling_y1, sampling_y2 + pad_h, patch_size)
        assert ((patch_xs[1:] - patch_xs[:-1]) == patch_size).all()
        assert ((patch_ys[1:] - patch_ys[:-1]) == patch_size).all()

        return patch_xs, patch_ys

    return (get_patch_coords,)


@app.cell
def _(np):
    def get_region(slide, coord, size, level=0):
        return np.array(slide.read_region(coord, level, size))

    return (get_region,)


@app.cell
def _(np):
    def has_content(image, threshold=0.02):
        green_ch = image[..., 1] / 255.0
        img_var = np.var(green_ch)
        num_blacks = (green_ch == 0).sum()

        return img_var > threshold and num_blacks < green_ch.size // 3

    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import tifffile
    import openslide

    import marimo as mo

    return mo, np, openslide, tifffile


if __name__ == "__main__":
    app.run()
