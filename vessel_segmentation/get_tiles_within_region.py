import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    image_dir_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="directory",
        multiple=False,
        label="Select the ***tiled images*** folder:"
    )

    mo.vstack([
        mo.md(r"""### Select the directory of the extracted tile images for species"""),
        mo.Html(
            r"""<p style='font-size: 90%'>for example: """
            r"""<i>.../cat bun 7579-2 side/anatomy/tiles/slide-2024-08-06T13-47-36-R1-S19/images</i></p><br>"""
        ),
        image_dir_browser
    ])
    return (image_dir_browser,)


@app.cell
def _(mo):
    geojson_browser = mo.ui.file_browser(
        initial_path="..",
        selection_mode="file",
        filetypes=[".geojson"],
        multiple=False,
        label="Select the ***GeoJSON*** file:"
    )

    mo.vstack([
        mo.md("### Select the extracted region *GeoJSON* file for the species"),
        geojson_browser
    ])
    return (geojson_browser,)


@app.cell
def _(geojson_browser, image_dir_browser):
    image_dir = None
    label_dir = None
    region_file = None

    if image_dir_browser.value:
        image_dir = image_dir_browser.path(0)
        label_dir = image_dir.parent / "labels"

    if geojson_browser.value:
        region_file = geojson_browser.path(0)
    return image_dir, label_dir, region_file


@app.cell
def _(Polygon, geojson, image_dir, mo, np, region_file):
    mo.stop(image_dir is None or region_file is None)

    region = geojson.load(region_file.open("r"))
    coords = np.array(region["features"][0]["geometry"]["coordinates"][0])

    region_polygon = Polygon(coords)
    region_polygon

    return (region_polygon,)


@app.cell
def _(image_dir, mo, region_polygon):
    mo.stop(region_polygon is None)

    # regular expression to extract x, y coordinates from tile names
    pattern = r".*\[x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]"

    # save directories
    save_dir = image_dir.parent.parent.parent / "tiles_within_region"
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_image_dir = save_dir / "images"
    saved_image_dir.mkdir(parents=True, exist_ok=True)

    saved_label_dir = save_dir / "labels"
    saved_label_dir.mkdir(parents=True, exist_ok=True)

    mo.md(f"##### Tiles within the region will be saved in *`{save_dir}`*")
    return pattern, saved_image_dir, saved_label_dir


@app.cell
def _(
    Point,
    image_dir,
    label_dir,
    mo,
    pattern,
    re,
    region_polygon,
    saved_image_dir,
    saved_label_dir,
    shutil,
):
    num_outside = 0
    num_inside = 0

    with mo.redirect_stdout():

        for img_file in image_dir.glob("*.tif"):
            match = re.match(pattern, img_file.stem)
            if not match:
                print(f"Skipping {img_file.name}, no match found.")
                continue
            x, y, w, h = map(int, match.groups())
            x2 = x + w
            y2 = y + h
            if region_polygon.contains(Point(x, y)) and region_polygon.contains(Point(x2, y2)):
                # print(f"Tile {img_file.stem} is within the region.")
                # copy the image and label to the save directory
                shutil.copy(img_file, saved_image_dir / img_file.name)
                label_file = label_dir / img_file.name
                shutil.copy(label_file, saved_label_dir / label_file.name)
                num_inside += 1
            else:
                print(f"{img_file.stem} is outside the region.")
                num_outside += 1
    
        print(f"\nNumber of tiles inside the region: {num_inside}")
        print(f"Number of tiles outside the region: {num_outside}")
    return


@app.cell
def _():
    import re
    import shutil
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import geojson

    from shapely.geometry import Point, Polygon
    return Point, Polygon, geojson, mo, np, re, shutil


if __name__ == "__main__":
    app.run()
