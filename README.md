# Planet_coding_test

The team has deployed a machine learning model to detect land use and land cover (LULC)
classes. For a 4 band   64 pixel   64 pixel image patch, the model provides a multiclass
prediction of the LULC types contained in the patch. After some time, the team notices that the
model does a poor job of differentiating between river (label = 1) and road (label = 2) classes.
To remedy this, you must provide additional training and testing patches to the Machine
Learning Engineers to help them fine-tune the model. To this end, you have been provided an
image with a shape of 4 bands   2,000 pixels   2,000 pixels that contains plentiful rivers and
roads, among other LULC labels. There is a corresponding image of pixelwise labels with a
single band and 2,000 pixels   2,000 pixels. 

Your task is to write a function that:
1. reads the image and labels, the filenames of which are provided to the function as input;
2. creates an appropriate sampling grid of the image data;
3. creates multiclass labels as the count of each label in a grid cell;
4. ignores grid cells that do not contain one of the two target classes;
5. splits the cells and their labels into training (~80%) and test (~20%) sets;
6. applies bandwise zero-centering and scaling against some constants BANDWISE_MEAN and
BANDWISE_STD , respectively (these can be assumed to be global variables that are arrays
of the same length as the number of spectral bands, i.e. four); and
7. returns the training and test sets and their respective labels.

For this exercise you may work in the language of your choice with a preference for Python. The
use of common libraries (e.g. rasterio and numpy in Python) is encouraged as long as the
code reflects an understanding of the underlying concepts of data preparation for machine
learning and does not, for example, leverage a library function to complete an entire step of the
exercise.
