import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""#### Select the image patch dataset directory (*for example .../species/fir sim 1125-1/dataset/level_2*)""")
    return


@app.cell
def _(mo):
    browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select the ***dataset level*** folder:",
    )

    browser
    return (browser,)


@app.cell
def _(mo):
    mo.md(r"""#### Select the FeatureForest predicted masks directory""")
    return


@app.cell
def _(mo):
    mask_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select the ***predicted masks*** folder:",
    )

    mask_browser
    return (mask_browser,)


@app.cell
def _(browser, get_patch_position, imread, mo, np):
    if not browser.value:
        mo.stop(True)

    dataset_folder = browser.path(0)
    patch_files = list(dataset_folder.glob("*.tiff"))
    patch_files.sort()
    # Load the first patch to get shape and dtype
    patch_img = imread(patch_files[0])
    patch_shape = patch_img.shape
    patch_dtype = patch_img.dtype

    # Get the patch positions
    patch_positions = [get_patch_position(pfile) for pfile in patch_files]
    num_rows, num_cols = np.max(patch_positions, axis=0) + 1

    print(f"Patch shape: {patch_shape}, dtype: {patch_dtype}")
    print(f"number of rows: {num_rows}, number of columns: {num_cols}")
    return (
        dataset_folder,
        patch_dtype,
        patch_files,
        patch_positions,
        patch_shape,
    )


@app.cell
def _(
    dask,
    get_dask_image,
    get_lazy_arrays,
    imread,
    patch_dtype,
    patch_files,
    patch_positions,
    patch_shape,
):
    lazy_imread = dask.delayed(imread)
    lazy_arrays = get_lazy_arrays(
        lazy_imread, patch_files, patch_shape, patch_dtype
    )

    chunk_shape = (3, 1024, 1024)

    dask_image = get_dask_image(
        lazy_arrays, patch_positions, patch_shape, patch_dtype
    )
    dask_image = dask_image.rechunk(chunk_shape)

    dask_image
    return chunk_shape, dask_image, lazy_imread


@app.cell
def _(
    get_dask_image,
    get_lazy_arrays,
    get_patch_position,
    lazy_imread,
    mask_browser,
    mo,
    np,
):
    if not mask_browser.value:
        mo.stop(True)

    # masks
    masks_folder = mask_browser.path(0)
    mask_files = list(masks_folder.glob("*.tiff"))
    mask_files.sort()

    mask_shape = (1, 512, 512)
    mask_dtype = np.uint8

    lazy_masks = get_lazy_arrays(
        lazy_imread, mask_files, mask_shape, mask_dtype
    )

    mask_positions = [get_patch_position(mfile) for mfile in mask_files]

    dask_mask = get_dask_image(
        lazy_masks, mask_positions, mask_shape, mask_dtype
    )
    dask_mask = dask_mask.rechunk((1, 1024, 1024))

    dask_mask
    return (dask_mask,)


@app.cell
def _(mo):
    get_allow_run, set_allow_run = mo.state(False)

    run_button = mo.ui.button(
        value=get_allow_run(),
        on_change=lambda value: set_allow_run(True),
        label="Save Zarr Dataset and Tiff Mask",
        kind="success",
    )

    run_button
    return get_allow_run, set_allow_run


@app.cell
def _(dask_mask, dataset_folder, get_allow_run, mo, tifffile):
    mo.stop(not get_allow_run())

    # load and save the large mask as a tiff image
    large_mask = dask_mask.compute()
    print(large_mask.shape)

    tifffile.imwrite(
        dataset_folder.parent.joinpath(f"dataset_{dataset_folder.name}_mask.tiff"),
        large_mask[0],
        photometric="minisblack",
        compression="zlib",
    )
    return


@app.cell
def _(
    Scaler,
    chunk_shape,
    dask_image,
    dataset_folder,
    get_allow_run,
    mo,
    parse_url,
    set_allow_run,
    uuid,
    write_image,
    zarr,
):
    mo.stop(not get_allow_run())

    # Create the zarr storage
    # the zarr dataset needs a new folder and cannot be overwritten 
    zarr_output = dataset_folder.parent.joinpath(f"{dataset_folder.name}.zarr")

    if zarr_output.exists():
        print("exists!")
        _name = f"{zarr_output.name}_old_{uuid.uuid4().time_low}"
        zarr_output.rename(zarr_output.parent.joinpath(_name))
        zarr_output = dataset_folder.parent.joinpath(f"{dataset_folder.name}.zarr")

    assert not zarr_output.exists()
    zarr_output.mkdir(parents=True, exist_ok=False)

    store = parse_url(zarr_output, mode="w").store
    root = zarr.group(store=store)
    root.attrs["omero"] = {
        "channels": [
            {
                "color": "FF0000",
                "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                "label": "red",
                "active": True,
            },
            {
                "color": "00FF00",
                "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                "label": "green",
                "active": True,
            },
            {
                "color": "0000FF",
                "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                "label": "blue",
                "active": True,
            }
        ]
    }

    scaler = Scaler(downscale=2, method="nearest")

    # write the dataset to the disk
    write_image(
        image=dask_image, 
        group=root, 
        axes="cyx", 
        storage_options=dict(chunks=chunk_shape),
        scaler=scaler,
    )

    # reset the button state/value
    set_allow_run(False)

    root.attrs.asdict()
    return


@app.cell
def _(mo):
    mo.md(r"""### Utility Functions""")
    return


@app.cell
def _(Path, da, np, tifffile):
    def get_patch_position(fname):
        # pattern: row_000_col_000.tiff
        parts = Path(fname).stem.split('_')
        row = int(parts[1])
        col = int(parts[3])

        return row, col


    def imread(fname):
        with tifffile.TiffFile(fname) as tif:
            img = tif.asarray()
            is_rgb = img.shape[-1] in (3, 4)
        if is_rgb:
            # convert to (C, H, W) format
            img = img.transpose((2, 0, 1))
        else:
            # convert to (1, H, W) format
            img = img[np.newaxis, ...]

        return img


    def get_lazy_arrays(lazy_imread, img_files, data_shape, dtype):
        lazy_images = [
            lazy_imread(imgf) for imgf in img_files
        ]
        # create a dask array from the lazy arrays
        lazy_arrays = [
            da.from_delayed(arr, shape=data_shape, dtype=dtype)
            for arr in lazy_images
        ]

        return lazy_arrays


    def get_dask_image(
        arrays, positions, data_shape, dtype
    ):
        # each array shape is C, H, W
        img_blocks = []
        num_rows, num_cols = np.max(positions, axis=0) + 1
        for row in range(num_rows):
            row_blocks = []
            for col in range(num_cols):
                # idx = row * num_cols + col
                try:
                    idx = positions.index((row, col))
                    row_blocks.append(arrays[idx])
                except ValueError:
                    # Fill with zeros if no patch exists for this position
                    row_blocks.append(da.zeros(data_shape, dtype=dtype))
                
            img_blocks.append(da.concatenate(row_blocks, axis=2))

        return da.concatenate(img_blocks, axis=1)



    return get_dask_image, get_lazy_arrays, get_patch_position, imread


@app.cell
def _():
    import uuid
    from pathlib import Path

    import numpy as np
    import tifffile
    import zarr
    import dask
    import dask.array as da
    import marimo as mo

    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image
    from ome_zarr.scale import Scaler
    return (
        Path,
        Scaler,
        da,
        dask,
        mo,
        np,
        parse_url,
        tifffile,
        uuid,
        write_image,
        zarr,
    )


if __name__ == "__main__":
    app.run()
