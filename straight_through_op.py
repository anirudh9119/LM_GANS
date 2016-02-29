import theano
import theano.tensor as T

import numpy.random as rng

class StraightThroughSampler(theano.Op):

    __props__ = ()

    itypes = [theano.tensor.dmatrix]
    otypes = [theano.tensor.ivector]

    def perform(self, node, inputs, outputs_storage):
        x = inputs[0]
        y = outputs_storage[0]
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        sample = srng.binomial(n=1, p=x, size=x.shape)

        y[0] = sample


    def grad(self, inputs, g):
        return [g[0]]

if __name__ == "__main__":

    sts = StraightThroughSampler()

    x = T.dmatrix()
    y = sts(x)

    #g = T.grad(T.sum(y),x)

    f = theano.function([x], outputs = {'y' : y})

    print f([[0.9, 0.5, 0.3, 0.1]])



