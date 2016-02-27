import io
from itertools import count

import numpy

from fuel.datasets.text import TextFile
from fuel.transformers import Merge
from fuel.schemes import ConstantScheme
from fuel.transformers import (Batch, Cache, Mapping, SortMapping, Padding,
                               Filter, Transformer)


EOS_TOKEN = '<EOS>'  # 0
UNK_TOKEN = '<UNK>'  # 1


class Shuffle(Transformer):
    def __init__(self, data_stream, buffer_size, **kwargs):
        if kwargs.get('iteration_scheme') is not None:
            raise ValueError
        super(Shuffle, self).__init__(
                data_stream, produces_examples=data_stream.produces_examples,
                **kwargs)
        self.buffer_size = buffer_size
        self.cache = [[] for _ in self.sources]

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        if not self.cache[0]:
            self._cache()
        return tuple(cache.pop() for cache in self.cache)

    def _cache(self):
        temp_caches = [[] for _ in self.sources]
        for i in range(self.buffer_size):
            for temp_cache, data in zip(temp_caches,
                                        next(self.child_epoch_iterator)):
                temp_cache.append(data)
        shuffled_indices = numpy.random.permutation(self.buffer_size)
        for i in shuffled_indices:
            for temp_cache, cache in zip(temp_caches, self.cache):
                cache.append(temp_cache[i])


def _source_length(sentence_pair):
    """Returns the length of the second element of a sequence.

    This function is used to sort sentence pairs by the length of the
    target sentence.

    """
    return len(sentence_pair[1])


def load_dict(filename, n_words=0):
    """Load vocab from TSV with words in last column."""
    dict_ = {EOS_TOKEN: 0, UNK_TOKEN: 1}
    with io.open(filename) as f:
        if n_words > 0:
            indices = range(len(dict_), n_words)
        else:
            indices = count(len(dict_))
        dict_.update(zip(map(lambda x: x.split()[-1], f), indices))
    return dict_


def get_stream(source, target, source_dict, target_dict, batch_size,
               buffer_multiplier=100, n_words_source=0, n_words_target=0,
               max_src_length=None, max_trg_length=None):
    """Returns a stream over sentence pairs.

    Parameters
    ----------
    source : list
        A list of files to read source languages from.
    target : list
        A list of corresponding files in the target language.
    source_dict : str
        Path to a tab-delimited text file whose last column contains the
        vocabulary.
    target_dict : str
        See `source_dict`.
    batch_size : int
        The minibatch size.
    buffer_multiplier : int
        The number of batches to load, concatenate, sort by length of
        source sentence, and split again; this makes batches more uniform
        in their sentence length and hence more computationally efficient.
    n_words_source : int
        The number of words in the source vocabulary. Pass 0 (default) to
        use the entire vocabulary.
    n_words_target : int
        See `n_words_source`.

    """
    if len(source) != len(target):
        raise ValueError("number of source and target files don't match")

    # Read the dictionaries
    dicts = [load_dict(source_dict, n_words=n_words_source),
             load_dict(target_dict, n_words=n_words_target)]

    # Open the two sets of files and merge them
    streams = [
        TextFile(source, dicts[0], bos_token=None,
                 eos_token=EOS_TOKEN).get_example_stream(),
        TextFile(target, dicts[1], bos_token=None,
                 eos_token=EOS_TOKEN).get_example_stream()
    ]
    merged = Merge(streams, ('source', 'target'))

    # Filter sentence lengths
    if max_src_length or max_trg_length:
        def filter_pair(pair):
            src, trg = pair
            src_ok = (not max_src_length) or len(src) < max_src_length
            trg_ok = (not max_trg_length) or len(trg) < max_trg_length
            return src_ok and trg_ok
        merged = Filter(merged, filter_pair)

    # Batches of approximately uniform size
    large_batches = Batch(
        merged,
        iteration_scheme=ConstantScheme(batch_size * buffer_multiplier)
    )
    sorted_batches = Mapping(large_batches, SortMapping(_source_length))
    batches = Cache(sorted_batches, ConstantScheme(batch_size))
    shuffled_batches = Shuffle(batches, buffer_multiplier)
    masked_batches = Padding(shuffled_batches)

    return masked_batches
