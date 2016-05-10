import tables
from fuel.datasets.hdf5 import PytablesDataset
from utils import make_local_copy

class TIMIT(PytablesDataset):
    """HMM-based TIMIT dataset.

    Parameters
    ----------
    which_set : str, opt
        either 'train', 'dev' or 'test'.
    """

    def __init__(self, which_set='train', local_copy=False):
        self.path = '/data/lisatmp4/speech/TIMIT/timit_framewise.h5'
        if local_copy and not self.path.startswith('/Tmp'):
            self.path = make_local_copy(self.path)
        self.which_set = which_set
        self.sources = ('features', 'features_shapes',
                        'labels', 'labels_shapes')
        super(TIMIT, self).__init__(
            self.path, self.sources, data_node=which_set)

    def get_normalization_factors(self):
        means = self.h5file.root._v_attrs.means
        stds = self.h5file.root._v_attrs.stds
        return means, stds
