'''
Build a simple neural language model using GRU units
'''
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy

import os



from data_iterator import TextIterator
from utils import  load_params, itemlist, init_tparams
import optimizers


from lm_base import (init_params, build_model, build_sampler,
                     gen_sample)

profile = False

def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of GRU units
          encoder='gru',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 weight decay penalty
          lrate=0.01,
          n_words=100000,  # vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          dataset='/data/lisatmp4/anirudhg/wiki.tok.txt.gz',
          valid_dataset='/data/lisatmp4/anirudhg/newstest2011.en.tok',
          dictionary='/data/lisatmp4/anirudhg/wiki.tok.txt.gz.pkl',
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    # load dictionary
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)

    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    # create shared variables for parameters
    tparams = init_tparams(params)

    # build the symbolic computational graph
    trng, use_noise, \
        x, x_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask]

    print 'Buliding sampler'
    f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams,
                                                             grads, inps, cost)

    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    f = open('generated_text.txt', 'w')
    # Training loop
    # generate some samples with the model and display them
    for jj in xrange(100000):
        sample, score = gen_sample(tparams, f_next,
                                   model_options, trng=trng,
                                   maxlen=30, argmax=False)
        print 'Sample ', jj, ': ',
        ss = sample
        for vv in ss:
            if vv == 0:
                break
            if vv in worddicts_r:
                print worddicts_r[vv],
                f.write(worddicts_r[vv])
                f.write(' ')
            else:
                print 'UNK',
                print
                f.write('UNK')
                f.write(' ')

        print '\n'
        f.write('\n')

    f.close()



if __name__ == '__main__':
    pass
