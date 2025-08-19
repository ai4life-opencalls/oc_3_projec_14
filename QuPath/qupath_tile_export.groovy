import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name)
mkdirs(pathOutput)

// Define output resolution
double requestedPixelSize = 1

// Convert to downsample
//double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = 1.0

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
    .downsample(downsample)               // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('vessels.and.small', 1)     // Choose output labels (the order matters!)
    .multichannelOutput(false)            // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)       // Define export resolution
    .imageExtension('.tif')       // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .labeledImageExtension('tif')
    .imageSubDir('images')
    .labeledImageSubDir('labels')
    .tileSize(512)                // Define size of each tile, in pixels
    // .useROIBounds(true)
    .labeledServer(labelServer)   // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)     // If true, only export tiles if there is a (labeled) annotation present
    .overlap(0)                   // Define overlap, in pixel units at the export resolution
    .exportJson(true)
    .writeTiles(pathOutput)       // Write tiles to the specified directory

print 'Done!'
