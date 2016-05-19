import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor
from utils import get_encoding

def gen_sample(dist, argmax=False, label_dim=181,
               rng=RandomStreams(2016)):
    rng = rng
    if not argmax:
        sample = rng.multinomial(pvals=dist)
    else:
        ind = tensor.argmax(dist, axis=1)
        sample = get_encoding(ind, label_dim)
    return sample.astype('int32')

