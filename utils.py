import hashlib
import numpy as np
import theano
import os
import shutil
from collections import OrderedDict

import theano.tensor as T
from theano.tensor.nnet import (binary_crossentropy,
                                categorical_crossentropy)

from fuel.streams import DataStream
from fuel.transformers import (Mapping, ForceFloatX, Padding,
                               SortMapping, Cast)
from fuel.schemes import ShuffledScheme
from datasets.schemes import SequentialShuffledScheme
from datasets.transformers import (MaximumFrameCache, Transpose, Normalize,
                                   AddUniformAlignmentMask, WindowFeatures,
                                   Reshape, AlignmentPadding, Subsample,
                                   ConvReshape)


floatX = theano.config.floatX

def make_local_copy(filename):
    local_name = os.path.join('/Tmp/', os.environ['USER'],
                              os.path.basename(filename))
    if (not os.path.isfile(local_name) or
                file_hash(open(filename)) != file_hash(open(local_name))):
        print '.. made local copy at', local_name
        shutil.copy(filename, local_name)
    return local_name

def get_top(array, N, states, cells, last_traces=None, choose='argmax'):
    if array.ndim == 2:
        if choose == 'argmax':
            top_labels = np.argsort(array, axis=1)[..., ::-1][..., :N]

        if not last_traces:
            label_list = top_labels.tolist()
            new_label_list = []
            for batch in label_list:
                new_batch = []
                for i in batch:
                    new_batch.append([i])
                new_label_list.append(new_batch)
            traces = new_label_list


    elif array.ndim == 3:
        array = array.transpose((1, 0 ,2)).reshape((array.shape[1], -1))
        if choose == 'argmax':
            top_labels = np.argsort(array, axis=1)[..., ::-1][..., :N]

        label_row = top_labels // 181
        label_col = top_labels % 181

        new_label_list = []
        for i in xrange(top_labels.shape[0]):
            new_batch = []
            for j in xrange(top_labels.shape[1]):
                new_batch.append(last_traces[i][label_row[i, j]] +
                                 [label_col[i, j]])
            new_label_list.append(new_batch)
        traces = new_label_list

    stack_label = []
    stack_prob = []

    stack_state = []
    stack_cell = []

    for i in xrange(N):
        indices = np.zeros((top_labels.shape[0], 181))
        if not last_traces:
            indices[np.arange(top_labels.shape[0]), top_labels[:, i]] = 1
        else:
            indices[np.arange(top_labels.shape[0]), label_col[:, i]] = 1
        prob = array[np.arange((top_labels.shape[0])), top_labels[:, i]]
        if not last_traces:
            sts = states
            cls = cells
        else:
            sts = states[label_row[:, i],np.arange((top_labels.shape[0]))]
            cls = cells[label_row[:, i], np.arange((top_labels.shape[0]))]
        stack_label.append(indices[None, ...])
        stack_prob.append(prob[None, ...])
        stack_state.append(sts[None, ...])
        stack_cell.append(cls[None, ...])

    return (traces,
            np.vstack(tuple(stack_label)).astype('int32'),
            np.vstack(tuple(stack_prob)).astype('float32'),
            np.vstack(tuple(stack_state)).astype('float32'),
            np.vstack(tuple(stack_cell)).astype('float32'))


def beam_search(func, data, data_mask, labels, state_out, cell_out,
                probs_out, beam_size=10, last_traces=None,
                choose='argmax'):

    all_states = []
    all_cells = []
    all_soft_outs = []

    for i in xrange(beam_size):
        soft_out, state_out_, cell_out_ = func(data, data_mask, labels[i],
                                               state_out[i], cell_out[i])
        all_soft_outs.append(soft_out * probs_out[i][:, None])
        all_states.append(state_out_)
        all_cells.append(cell_out_)

    all_states = np.asarray(all_states)
    all_cells = np.asarray(all_cells)
    all_soft_outs = np.asarray(all_soft_outs)
    return get_top(all_soft_outs, beam_size, all_states,
                   all_cells, last_traces=last_traces, choose=choose)


def get_encoding(targets, dim):
    shape = targets.shape
    ndim = targets.ndim
    if ndim == 2:
        targets = targets.flatten()
    zeros_targets = T.zeros((targets.shape[0], dim))
    new_targets = T.set_subtensor(zeros_targets[T.arange
        (targets.shape[0]), targets], 1)
    if ndim == 2:
        return new_targets.reshape((shape[0],
                                    shape[1], dim),
                                   ndim=3).astype('int32')
    else:
        return new_targets.reshape((shape[0], dim),
                                   ndim=2).astype('int32')


def sequence_categorical_crossentropy(prediction, targets, mask):
    prediction_flat = prediction.reshape(((prediction.shape[0] *
                                           prediction.shape[1]),
                                          prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    ce = categorical_crossentropy(prediction_flat, targets_flat)
    return T.sum(ce * mask_flat)

def sequence_binary_crossentropy(prediction, targets, mask):
    prediction_flat = prediction.flatten()
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    ce = binary_crossentropy(prediction_flat, targets_flat)
    return T.sum(ce * mask_flat)

def sequence_binary_misrate(prediction, targets, mask):
    prediction_flat = prediction.flatten()#reshape(((prediction.shape[0] *
                                          # prediction.shape[1]),
                                          #prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    #neq = T.neq(T.argmax(prediction_flat, axis=1), targets_flat)
    neq = T.neq(prediction_flat, targets_flat)
    neq *= mask_flat
    use_length = mask_flat.sum()
    mr = T.sum(neq) / T.cast(use_length, 'floatX')
    return T.sum(mr)

def sequence_misclass_rate(prediction, targets, mask):
    prediction_flat = prediction.reshape(((prediction.shape[0] *
                                           prediction.shape[1]),
                                          prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    neq = T.neq(T.argmax(prediction_flat, axis=1), targets_flat)
    neq *= mask_flat
    use_length = mask_flat.sum()
    mr = T.sum(neq) / T.cast(use_length, 'floatX')
    return T.sum(mr)

class Normalizer(object):

    def __init__(self):
        self.sum = 0.0
        self.sum_of_squares = 0.0
        self.N = 0
        self.trained = False

    def fit(self, generator):
        iterator = generator()
        for x in iterator:
            self.sum += x.sum(0)
            self.N += x.shape[0]

        # separate ss pass for numerical stability
        iterator = generator()
        self.x_mean = self.sum / self.N
        for x in iterator:
            self.sum_of_squares += ((x - self.x_mean)**2).sum(0)

        self.x_stdev = np.sqrt(self.sum_of_squares / self.N)
        self.trained = True

    def apply(self, x):
        assert self.trained
        return (x - self.x_mean) / self.x_stdev


def key(x):
    return x[0].shape[0]


def construct_hmm_stream(dataset, rng, pool_size, maximum_frames, window_features,
                         **kwargs):
    """Construct data stream.

    Parameters:
    -----------
    dataset : Dataset
        Dataset to use.
    rng : numpy.random.RandomState
        Random number generator.
    pool_size : int
        Pool size for TIMIT dataset.
    maximum_frames : int
        Maximum frames for TIMIT datset.
    subsample : bool, optional
        Subsample features.
    pretrain_alignment : bool, optional
        Use phoneme alignment for pretraining.
    uniform_alignment : bool, optional
        Use uniform alignment for pretraining.

    """
    kwargs.setdefault('subsample', False)
    kwargs.setdefault('pretrain_alignment', False)
    kwargs.setdefault('uniform_alignment', False)
    stream = DataStream(
        dataset,
        iteration_scheme=SequentialShuffledScheme(dataset.num_examples,
                                                  pool_size, rng))
    stream = Reshape('features', 'features_shapes', data_stream=stream)
    stream = Reshape('labels', 'labels_shapes', data_stream=stream)
    means, stds = dataset.get_normalization_factors()
    stream = Normalize(stream, means, stds)
    if not window_features == 1:
        stream = WindowFeatures(stream, 'features', window_features)
    stream.produces_examples = False
    stream = Mapping(stream,
                     SortMapping(key=key))
    stream = MaximumFrameCache(max_frames=maximum_frames, data_stream=stream,
                               rng=rng)
    stream = Padding(data_stream=stream,
                     mask_sources=['features', 'labels'])
    stream = Transpose(stream, [(1, 0, 2), (1, 0), (1, 0), (1, 0)])

    stream = ForceFloatX(stream)
    if kwargs['subsample']:
        stream = Subsample(stream, 'features', 5)
        stream = Subsample(stream, 'features_mask', 5)
    return stream
