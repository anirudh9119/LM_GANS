'''
Build a simple neural language model using GAN Auxillary loss
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from straight_through_op_2 import straight_through
import numpy
from theano.ifelse import ifelse
import os


from layers import get_layer




profile = False

def save_params(params, filename, symlink=None):
    """Save the parameters.
       Saves the parameters as an ``.npz`` file. It optionally also creates a
       symlink to this archive.
    """
    numpy.savez(filename, **params)
    if symlink:
        if os.path.lexists(symlink):
            os.remove(symlink)
        os.symlink(filename, symlink)


# build a training model
def build_GAN_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    bern_dist = tensor.matrix('bern_dist', dtype='float32')
    uniform_sampling = tensor.vector('uniform_sampling', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # input
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    opt_ret['emb'] = emb

    # pass through gru layer, recurrence here
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    proj_h = proj[0]
    opt_ret['proj_h'] = proj_h

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    ind_max =  probs.argmax(1)

    one_hot_sampled = straight_through(probs, uniform_sampling)
    one_hot_input = one_hot_sampled.reshape([logit_shp[0],logit_shp[1], logit_shp[2]])
    print "probs ndim", probs.ndim
    print "uniform_sampling ndim", uniform_sampling.ndim


    temp_x = x.flatten()
    output_mask = bern_dist.flatten()


#   ind_new_mask = ifelse(tensor.eq(output_mask, 0),ind_max, temp_x)
#   emb = ifelse(tensor.eq(output_mask[0], 0), theano.dot(one_hot_sampled, tparams['Wemb']), tparams['Wemb'][temp_x])

    ones_vec = tensor.switch(tensor.lt(temp_x, 30000), 1, 0)
    ones_vec = tensor.cast(ones_vec, 'int32')

    cum_sum = tensor.extra_ops.cumsum(ones_vec)
    cum_sum = tensor.switch(tensor.gt(cum_sum, 0), cum_sum -1, 0)
    new_emb =  theano.dot(one_hot_sampled, tparams['Wemb'])


    disc_emb = new_emb[cum_sum]
    gen_emb = tparams['Wemb'][temp_x]
    emb = tensor.switch(tensor.eq(output_mask, 0), disc_emb.T, gen_emb.T)
    emb = emb.T

    f_get =  theano.function([x, x_mask, bern_dist, uniform_sampling],[probs, ind_max,one_hot_sampled,
                                                            ones_vec, cum_sum, new_emb, emb])


    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    opt_ret['emb'] = emb

    # pass through gru layer, recurrence here
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    proj_h = proj[0]
    opt_ret['proj_h'] = proj_h

    # compute word probabilities
    #TODO: turn lstm back on
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = 0.0 * get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')

    logit_lstm = logit_lstm# + 0.1 * trng.normal(size = logit_lstm.shape)

    logit = tensor.tanh(logit_lstm + 0.0 * logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    opt_ret['cost_per_sample'] = cost
    cost = (cost * x_mask).sum(0)

    return trng, use_noise, x, x_mask, opt_ret, cost, f_get, bern_dist, uniform_sampling, one_hot_input, get_layer("gru")[1](tparams, emb, options, prefix='encoder', mask=x_mask)[0], emb

# build a sampler
def build_GAN_sampler(tparams, options, trng):
    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][y])


    # apply one step of gru layer
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=None,
                                            one_step=True,
                                            init_state=init_state)
    next_state = proj[0]


    # compute the output probability dist and sample
    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')


    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')


    logit = tensor.tanh(logit_lstm + 0.0 * logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')



    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..',
    inps = [y, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_next, logit, next_state

