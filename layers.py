import theano
from theano import tensor
import numpy
from utils import uniform_weight, ortho_weight, norm_weight

import settings
profile = settings.profile


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
         }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def zero_vector(length):
    return numpy.zeros((length, )).astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x




# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options,
                       param,
                       prefix='ff',
                       nin=None,
                       nout=None,
                       ortho=True):

    param[prefix + '_W'] = uniform_weight(nin, nout)
    param[prefix + '_b'] = zero_vector(nout)

    return param


def fflayer(tparams,
            state_below,
            options,
            prefix='rconv',
            activ='lambda x: tensor.tanh(x)',
            **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[prefix + '_W']) +
                       tparams[prefix + '_b'])


# GRU layer
def param_init_gru(options, param, prefix='gru', nin=None, dim=None):

    param[prefix + '_W'] = numpy.concatenate(
        [
            uniform_weight(nin, dim), uniform_weight(nin, dim)
        ],
        axis=1)
    param[prefix + '_U'] = numpy.concatenate(
        [
            ortho_weight(dim), ortho_weight(dim)
        ],
        axis=1)
    param[prefix + '_b'] = zero_vector(2 * dim)

    param[prefix + '_Wx'] = uniform_weight(nin, dim)
    param[prefix + '_Ux'] = ortho_weight(dim)
    param[prefix + '_bx'] = zero_vector(dim)

    return param


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]

def _step_slice(m_, x_, xx_,  h_, U, Ux):
    preact = tensor.dot(h_, U)
    preact += x_

    # reset and update gates
    r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
    u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

    # compute the hidden state proposal
    preactx = tensor.dot(h_, Ux)
    preactx = preactx * r
    preactx = preactx + xx_

    # hidden state proposal
    h = tensor.tanh(preactx)

    # leaky integrate and obtain next hidden state
    h = u * h_ + (1. - u) * h
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h

def gru_layer(tparams,
              state_below,
              options,
              prefix='gru',
              mask=None,
              one_step=False,
              init_state=None,
              **kwargs):
    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[prefix + '_Ux'].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # state_below is the input word embeddings
    state_below_ = (tensor.dot(state_below, tparams[prefix + '_W']) +
                    tparams[prefix + '_b'])
    state_belowx = (tensor.dot(state_below, tparams[prefix + '_Wx']) +
                    tparams[prefix + '_bx'])

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[prefix + '_U'], tparams[prefix + '_Ux']]
    # set initial state to all zeros
    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:  # sampling
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state],
                                non_sequences=shared_vars,
                                name=prefix + '_layers',
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval
