import numpy
import numpy as np

def gen_sample_conditional(tparams, f_next, options, initial_text, worddicts, trng=None, maxlen=30, argmax=True):

    initial_text_indices = []

    for word in initial_text:
        if word in worddicts:
            nw = worddicts[word]
        else:
            nw = 0

        if nw >= 30000:
            nw = 1

        initial_text_indices.append(nw)


    sample = []
    sample_score = 0

    # initial token is indicated by a -1 and initial state is zero
    next_w = -1 * numpy.ones((1,)).astype('int64')
    next_state = numpy.zeros((1, options['dim'])).astype('float32')

    next_state_lst = []

    for ii in xrange(maxlen):
        inps = [next_w, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        next_state_lst += [next_state]

        if ii < len(initial_text_indices):
            nw = initial_text_indices[ii]
        elif argmax:
            nw = next_p[0].argmax()
        else:
            nw = next_w[0]

        sample.append(nw)
        sample_score += next_p[0, nw]
        if nw == 0:
            break

    return sample


