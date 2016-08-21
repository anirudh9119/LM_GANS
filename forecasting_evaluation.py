

from utils import zipp, unzip, init_tparams, load_params, itemlist
import os

from lm_base import (init_params, build_sampler,gen_sample, gen_sample_batch, pred_probs, prepare_data)

from conditional_sampler import gen_sample_conditional

import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import time

'''
Given examples, runs evaluations going forward for one year.  
This is "conditional sampling".  
We do this k times, and then we can compute metrics like MAPE and quantile loss.  

Try to determine proper amount of biasing.  

Also can sample farther ahead than is observed, which is interesting analytically.  

For example, how does the model project holidays like 30 years into the future?  

'''

text_file = open("/home/ubuntu/lambalex/PF_exp/datafile_train.txt", "r")

import matplotlib.pyplot as plt

def extract_feature(lst, feature):
    ex = []
    for e in lst:
        if feature + "_" in e:
            ex.append(float(e.split("_")[1]))
    return np.asarray(ex)


if __name__ == "__main__":

    options = {"n_words" : 2013, "dim_word" : 512, 'encoder' : 'gru', 'dim' : 512}

    params = {}#init_params(options)

    saved_model_loc = '/home/ubuntu/lambalex/PF_exp/logs/1471312679/1471312679.model.npz'

    dictionary = "/home/ubuntu/lambalex/PF_exp/dictionary.npz"

    import numpy.random as rng
    trng = RandomStreams(rng.randint(0,1000))

    params_n = load_params(saved_model_loc, params)

    print "LOADED PARAMS!"
    print "param keys", params_n.keys()

    # create shared variables for parameters
    tparams = init_tparams(params)


    a = np.load(dictionary)
    i2w = a['arr_0'][0]
    w2i = a['arr_0'][1]
    worddicts = w2i

    f_next = build_sampler(tparams, options, trng, biased_sampling_term = 1.0)

    import random
    for j in range(random.randint(10,200)):
        initial_text = text_file.readline().rstrip("\n").split(" ")[:1200]

    #initial_text = []

    num_samples = 10
    
    samples = []

    for i in range(num_samples):
        t0 = time.time()
        gs = gen_sample_conditional(tparams, f_next, options, initial_text, worddicts, trng=trng, maxlen=1200 + 52*6, argmax=False)
        print "time", time.time() - t0

        gst = []
        for i in range(len(gs)):
            gst.append(i2w[gs[i]])


        plt.axvline(len(initial_text) / 6.0, color = 'red', linewidth = 3.0)
        eas = extract_feature(gst, "eas")
        dmd = extract_feature(gst, "d")
        av = extract_feature(gst, "av")

        samples.append(dmd)

        print "------------------------------------------------------"

        print "av", av
        print "demand", dmd
        print "dmd shape", dmd.shape[0]

        plt.plot(dmd)
        plt.show()

    import numpy as np
    samples = np.vstack(samples)
    print samples.shape

    plt.plot(np.median(samples, axis = 0).tolist())

    plt.show()





