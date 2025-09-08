<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://github.com/ai4life-opencalls/.github/blob/main/AI4Life_banner_giraffe_nodes_OC.png?raw=true" width="70%">
  </a>
</p>

# Project #14: The speed of life in trees; Linking wood anatomy with wood lifespan and tree growth

In this project, scientist from the Hawkesbury Institute for the Environment, Western Sydney University wanted to investigate how wood anatomy is linked to wood lifespan and tree growth. There are four main cell types in wood: *Fiber*, *Vessel*, *Axial Parenchyma* and *Ray*. Each type of cell plays a different role in the transport of water and nutrients throughout the plant. By analyzing these cells, scientists hope to gain insights into how they affect wood lifespan and tree growth.  
The data for this project was aquired using brightfield microscopy (BFM) of cross-sections of wood samples, which allows researchers to capture images of the entire sample at once. This method provides high-resolution images that can be used to analyze individual cells within the wood structure. The data was included of 93 whole-slide images of 51 species.
<div>
<img src="assets/sp1.png" width="320px"/>
<img src="assets/sp2.png" width="320px"/>
<img src="assets/sp3.png" width="320px"/>
</div>  

## Pipeline Overview
To achieve the desired analysis of the project, as the first step we needed to apply a semantic segmentation model to segment the image into its respective cell types. But the provided data contained ground truth masks only for *Vessel* cells. Therefore, we provided two different approaches to solve this problem:
- For *Vessel* cells, we trained a deep learning model to achieve the segmentation masks.
- For other cell types (*Fiber*, *Axial Parenchyma* and *Ray*), we created a pipeline using [**FeatureForest**](https://github.com/juglab/featureforest), a napari plugin that uses a few user-provided scribble labels to generate semantic segmentation masks of desired cell types.


## Installation
First you need to install [conda] or [mamba] as your python package and dependency manager.  
We provided `install.sh` script for Linux & Mac OS users, and `install.bat` script for Windows users.  
First you need to clone the repo:
```bash
git clone https://github.com/ai4life-opencalls/oc_3_projec_14
cd ./oc_3_projec_14
```
Now run the installation script:
```bash
# Linux or Mac OS
sh ./install.sh
```
```bash
# Windows
./install.bat
```
This will install all required packages and dependencies including *napari* and the **FeatureForest** plugin. For more details about installation you can refer to the [installation page](https://juglab.github.io/featureforest/install/) of the **FeatureForest** plugin.


## Semantic Segmentation of *Vessel* Cells
to be completed...


## Generating masks for other cell types
to be completed...

---
<img src="assets/eu_flag.jpg" height="50" align="left" style="margin: 5px 10px 0 0">
<div style="text-align: justify">AI4Life has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.</div>


[conda]: https://conda.io/projects/conda/en/latest/index.html
[mamba]: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html