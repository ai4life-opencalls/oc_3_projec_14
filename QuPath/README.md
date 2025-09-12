## Export Annotated Tiles in QuPath

This [script](qupath_tile_export.groovy) is used to export annotated tiles from a QuPath project. To use this script, you need to open your QuPath project first and then load and run the script.  
To load the script, from the menu barclick on **Automate** -> **Script editor** and then copy/paste the code into the text box. Then click on **Run**.  You can also put the provided script file into the *user* or *shared* scripts folder of your QuPath installation directory.  

#### Script parameters:
- Export directory:  
`def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name)`  
This parameter defines where the exported tiles will be saved. The default value is set as `PROJECT_BASE_DIR/tiles/{image_name}`.

- Downsampling:  
`double downsample = 1.0`  
This parameter defines how much to scale down the tiles before exporting it. For example, if you want to export a tile at half resolution, set this parameter to `2`.  

- Annotation to export:  
`.addLabel('vessels.and.small', 1)`  
This parameter defines which annotations should be exported. In this case, we are only interested in `vessels.and.small` annotations and we set the class label to `1` as opposed to `0` for background. If you have other types of annotations that you would like to include, add them here with their corresponding class label ID.  

- Tile size:  
`.tileSize(512)`  
This parameter defines the size of each tile (in pixels). The default value is `512x512`.  
