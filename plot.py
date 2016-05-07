
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def plot_image_grid(images, nrows, ncols, save_path=None):

    images = np.clip(images, 36.0, 37.0)
    images = images - 36.0
    
    figure = pyplot.figure()
    grid = ImageGrid(figure, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1)

    if images[0].shape[0] == 1:
        cmap = cm.Greys_r
    else:
        cmap = cm.Greys_r

    for i, image in enumerate(images):
        grid[i].axis('off')
        grid[i].imshow(images[i].transpose(1, 2, 0).squeeze(),
                       cmap = cmap, interpolation='nearest')

    #cm.Greys_r

    pyplot.tight_layout(pad=0)
    if save_path is None:
        pyplot.show()
    else:
        pyplot.savefig(save_path, transparent=True, bbox_inches='tight')

