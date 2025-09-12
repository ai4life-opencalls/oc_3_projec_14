# Pipeline Description
Here we describe the pipeline for the segmentation of the *Vessel* cells. This includes the data preparation steps, and training/testing of the model.

## Pipeline Steps

### Step 1: Exporting Tiles
First we need to make a dataset from the annotated part of the whole-slide images. Each provided image came with a *QuPath* project that contains annotations on the vessels in a pie-shaped region. We can export tiles (patches) and their labels (masks) using *QuPath* `TileExporter`.  

<img src="../assets/vessel/qp_annotations.png" height="340px">  

To export these tiles and their masks, please check the script and documentation at [here](../QuPath/README.md). The exported tiles will be saved as `.tif` files under `tiles` directory including `images` and `labels`.

### Step 2: Tiles Filtering
Since the annotated regions are pie-shaped, some of tiles at the edges may contain only partial annotation which means they might contain more *vessels* but without annotations. This will make it hard for model to distinguish between *vessels* and *non-vessels*. Therefore, we need to filter out those tiles before feeding them into the model.  

To do so, first we need to export the polygon area around each annotated pie area. In QuPath, select the *vessel.region.small* annotation and export the polygon as a `GeoJSON` file.

<img src="../assets/vessel/qp_ex_geo_1.png" height="170px">

<img src="../assets/vessel/qp_ex_geo_2.png" height="220px">

<img src="../assets/vessel/qp_ex_geo_3.png" height="204px">

<br>

After that, you can run `get_tiles_within_region.py` notebook to do this filtering step:
```bash
marimo run get_tiles_within_region.py
```
