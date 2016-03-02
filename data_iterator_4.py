from itertools import count

from fuel.datasets.text import TextFile
from fuel.transformers import Merge
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Cache, Mapping, SortMapping, Filter, Padding



EOS_TOKEN = '<EOS>'  # 0
UNK_TOKEN = '<UNK>'  # 1


def _source_length(sentence_pair):
    """Returns the length of the first element of a sequence.

    This function is used to sort sentence pairs by the length of the
    source sentence.

    """
    return len(sentence_pair[0])

def load_dict(filename, n_words=0):
    """Load vocab from TSV with words in last column."""
    dict_ = {EOS_TOKEN: 0, UNK_TOKEN: 1}
    with open(filename) as f:
        if n_words > 0:
            indices = range(len(dict_), n_words)
        else:
            indices = count(len(dict_))
        dict_.update(zip(map(lambda x: x.split()[-1], f), indices))
    return dict_





def get_stream(source, source_dict, batch_size=128, buffer_multiplier=100, n_words_source=0, max_src_length=None, max_trg_length=None):
    """Returns a stream over sentence pairs.

    Parameters
    ----------
    source : list
        A list of files to read source languages from.
    source_dict : str
        Path to a tab-delimited text file whose last column contains the
        vocabulary.
    batch_size : int
        The minibatch size.
    buffer_multiplier : int
        The number of batches to load, concatenate, sort by length of
        source sentence, and split again; this makes batches more uniform
        in their sentence length and hence more computationally efficient.
    n_words_source : int
        The number of words in the source vocabulary. Pass 0 (default) to
        use the entire vocabulary.

    """
    # Open the two sets of files and merge them
    dicts = [load_dict(source_dict, n_words=n_words_source)]

    streams = [
        TextFile(source, dicts[0], bos_token=None,
                 eos_token=EOS_TOKEN).get_example_stream(),
        TextFile(source, dicts[0], bos_token=None,
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
    masked_batches = Padding(batches)

    return masked_batches
