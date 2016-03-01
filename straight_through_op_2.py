import theano
import theano.tensor as T

import numpy as np


class StraightThroughSampler(theano.Op):

    __props__ = ()

    itypes = [theano.tensor.fmatrix, theano.tensor.fmatrix]
    otypes = [theano.tensor.fmatrix]

    def perform(self, node, inputs, outputs_storage):
        y = outputs_storage[0]

        y[0] = inputs[1]

    def grad(self, inputs, g):

        one_hot = inputs[1]

        g1 = g[0]

        #input is 2 x 4.  Output is 2 x 1.

        g2 = g1 * one_hot

        return [g2, T.cast(T.ones_like(inputs[1]), 'float32')]

def straight_through(p, u):


    sts = StraightThroughSampler()

    raw_cum = T.extra_ops.cumsum(p, axis = 1) - T.addbroadcast(T.reshape(u, (u.shape[0], 1)),1)

    cum = T.switch(T.lt(raw_cum, 0.0), 10.0, raw_cum)
    ideal_bucket = T.argmin(cum, axis = 1)
    one_hot = T.extra_ops.to_one_hot(ideal_bucket, 30000)
    y = sts(p, one_hot)

    return y

if __name__ == "__main__":

    p = T.fmatrix()
    u = T.fvector()

    f = theano.function([p, u], outputs = {'st' : straight_through(p,u)})

    mat = np.load('outfile.npz')

    print f(mat['arr_0'], mat['arr_1'])

    raise Exception("DONE")

    p = T.fmatrix()
    u = T.fvector()

    y = straight_through(p,u)

    g = T.grad(T.sum(y**2), p)

    g2 = T.grad(T.sum(p**2), p)

    f = theano.function([p, u], outputs = {'y' : y, 'g' : g, 'g2' : g2})

    unif = np.asarray([0.5, 0.5]).astype('float32')

    r = f(np.asarray([[0.01, 0.0, 0.99, 0.0],[0.1,0.4,0.2,0.3]]).astype('float32'), unif)

    print "sampled"
    print r['y']

    print "grad"
    print r['g']

    print 'g2'
    print r['g2']


