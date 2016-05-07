'''
Build a simple neural language model using GRU units
So on each time you have a matrix of probabilities, p
p is a 64 x 30k matrix if we have 64 examples and 30k words
your other input is the word embeddings
either for the discriminator on the generator on the next step
your op should sample from a multinomial correspond to p,
grab the right word embeddings, and return them
'''
from plot import plot_image_grid
import logging
import theano
import theano.tensor as tensor
import lasagne
import random
from PIL import Image

theano.config.scan.allow_gc = True

import cPickle as pkl
import ipdb
import numpy
import copy
from toolz.dicttoolz import merge
from images2gif import writeGif

from consider_constant import consider_constant

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import time

from platoon.channel import Worker
from mimir import RemoteLogger

from six.moves import xrange
from char_data_iterator import TextIterator
#from data import  char_sequence
from utils import zipp, unzip, init_tparams, load_params, itemlist
import optimizers

#from Descriminator_Hidden import discriminator
from Discriminator_Conv import discriminator

from lm_base import (init_params, build_sampler,gen_sample, gen_sample_batch, pred_probs, prepare_data)

from lm_discriminator import  build_GAN_model, save_params

#from conditional_sampler import gen_sample_conditional
logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
LOGGER = logging.getLogger(__name__)

profile = False

from sklearn.manifold import TSNE

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
#from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.manifold.t_sne import (_kl_divergence) #_joint_probabilities
#from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
#import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                 rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
#from moviepy.video.io.bindings import mplfig_to_npimage
#import moviepy.editor as mpy

def scatter(x, colors):
     # We choose a color palette with seaborn.
     palette = numpy.array(sns.color_palette("hls", 10))

     # We create a scatter plot.
     f = plt.figure(figsize=(8, 8))
     ax = plt.subplot(aspect='equal')
     sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                     c=palette[colors.astype(numpy.int)])
     plt.xlim(-25, 25)
     plt.ylim(-25, 25)
     ax.axis('off')
     ax.axis('tight')

     # We add the labels for each digit.
     txts = []
     for i in range(2):
         # Position of each label.
         xtext, ytext = numpy.median(x[colors == i, :], axis=0)
         txt = ax.text(xtext, ytext, str(i), fontsize=24)
         txt.set_path_effects([
             PathEffects.Stroke(linewidth=5, foreground="w"),
             PathEffects.Normal()])
         txts.append(txt)

     return f, ax, sc, txts



def train(worker, model_options, data_options,
          dim_word,  # word vector dimensionality
          dim,  # the number of GRU units
          encoder,
          patience,  # early stopping patience
          max_epochs,
          finish_after,  # finish after this many updates
          dispFreq,
          decay_c,  # L2 weight decay penalty
          lrate,
          n_words,  # vocabulary size
          maxlen,  # maximum length of the description
          minlen,
          optimizer,
          batch_size,
          valid_batch_size,
          saveto,
          validFreq,
          saveFreq,  # save the parameters after every saveFreq updates
          sampleFreq,  # generate some samples after every sampleFreq
          dataset,
          valid_dataset,
          dictionary,
          use_dropout,
          reload_,
          train_generator_flag,
          batch_port,
          log_port,
          control_port,
          save_tsne,
          flag_save_tsne,
          use_gan_objective,
          model_name,
          beta1,
          beta2,
          learning_rate,
          limit_desc_start,
          limit_desc_end,
          gan_starting_point):

    LOGGER.info('Connecting to data socket ({}) and loading validation data'
                    .format(batch_port))
    worker.init_mb_sock(batch_port)
    log = RemoteLogger(port=log_port)


    experiment_id = worker.send_req('experiment_id')
    model_filename = '{}.model.npz'.format(experiment_id)
    saveto_filename = '{}.npz'.format(saveto)

    try:
        os.mkdir(saveto + experiment_id)
    except:
        pass

    saveto += experiment_id

    # Model options
    model_options = locals().copy()

    '''
    # load dictionary
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    #worddicts: word -> index
    #worddicts_r : index -> word.
'''
    # reload options
    #if reload_ and os.path.exists(saveto):
    #    with open('%s.pkl' % saveto, 'rb') as f:
    #        model_options = pkl.load(f)


    LOGGER.info('Loading data')
    import numpy
    a = numpy.load(dictionary)
    i2w = a['arr_0'][0]
    w2i = a['arr_0'][1]
    worddicts = w2i
    worddicts_r = i2w
    '''
    train, i2w, w2i, data_xy = char_sequence(i2w, w2i, f_path = dataset,
                                             batch_size = batch_size)
    valid, i2w, w2i, data_xy = char_sequence(i2w, w2i, f_path = valid_dataset,
                                             batch_size = valid_batch_size)
    '''
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

    LOGGER.info('Building model')
    params = init_params(model_options)

    # reload parameters
    #saved_model_loc = "/u/lambalex/logs/mnist/1462399235/1462399235.model.npz"
    #saved_model_loc = '/u/lambalex/logs/mnist/1462407058/1462407058.model.npz'
    saved_model_loc = '/u/lambalex/logs/mnist/1462511667/1462511667.model.npz'

    if reload_ and os.path.exists(saveto):
        params = load_params(saved_model_loc, params)

    # create shared variables for parameters
    tparams = init_tparams(params)

    # build the symbolic computational graph

    trng, use_noise,\
        x, x_mask,\
        opt_ret,\
        cost,\
        bern_dist,\
        uniform_sampling,\
        hidden_states, emb_obs, get_hidden, probs_tf = build_GAN_model(tparams, model_options)

    trng_sampled, \
        use_noise_sampled, \
        x_sampled, x_mask_sampled, \
        opt_ret_sampled, cost_sampled,\
        bern_dist_sampled, uniform_sampling_sampled, \
        hidden_states_sampled, emb_sampled, get_hidden_sampled, probs_rf = build_GAN_model(tparams, model_options)

    #MMD score

    #h_tf_mean = tensor.specify_shape(tensor.mean(hidden_states, axis = 1), (785, 1024 + 50))
    #h_rf_mean = tensor.specify_shape(tensor.mean(hidden_states_sampled, axis = 1), (785, 1024 + 50 + 99))
    #mmd_score = tensor.mean((h_tf_mean - h_rf_mean)**2)


    hidden_states = consider_constant(hidden_states)

    hidden_states_joined = tensor.concatenate([hidden_states, hidden_states_sampled], axis = 1)

    inps = [x, x_mask, bern_dist, uniform_sampling]
    inps_sampled = [x_sampled, x_mask_sampled, bern_dist_sampled, uniform_sampling_sampled]


    LOGGER.info('Building sampler')
    b_sample_term = 1.0
    log.log({'sampling bias term' : b_sample_term})
    f_next = build_sampler(tparams, model_options, trng, b_sample_term)

    # before any regularizer
    LOGGER.info('Building f_log_probs')
    f_log_probs = theano.function(inps, cost, profile=profile)
    LOGGER.info('Building f_log_probs Done')


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
    LOGGER.info('Building f_cost')
    f_cost = theano.function(inps, cost, profile=profile)
    LOGGER.info('Done')

    LOGGER.info('Computing gradient')
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    LOGGER.info('Done')

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    LOGGER.info('Building optimizers')
    f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams,
                                                             grads, inps, cost)




    LOGGER.info('Done')


    history_errs = []
    # reload history
    #if reload_ and os.path.exists(saveto):
    #    history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0

    discriminator_target = tensor.ivector()


    d = discriminator(num_hidden = 2048,
                      num_features = 512 + 50,
                      seq_length = maxlen, mb_size = batch_size,
                      hidden_state_features = hidden_states_joined,
                      target = discriminator_target)

    last_acc = 0.0
    discriminator_accuracy_moving_average = 0.0

    '''
        -Get two update functions:
          -Update generator wrt. disc.
          -Update discriminator wrt. generator outputs and real outputs.
        -Use the same inputs for both.
    '''

    

    #target = 1 corresponds to teacher forcing, target = 0 corresponds to sampled sentences.

    print "tparam keys"
    print tparams.keys()

    if use_gan_objective:
        tparams_gen = []

        tparams_gen = tparams.values()

        '''
        generator_loss = tensor.mean(-d.loss * (1.0 - discriminator_target))
        generator_gan_updates = lasagne.updates.adam(tensor.cast(generator_loss, 'float32'),
                                                 tparams_gen, learning_rate = 0.0001,
                                                 beta1 = 0.5)
        discriminator_gan_updates = lasagne.updates.adam(tensor.mean(d.loss),
                                                         d.params, learning_rate = 0.0001,
                                                         beta1 = 0.5)
        '''

        #disc_lr_mult = tensor.scalar()
        #gen_lr_mult = tensor.scalar()

        g_grads = lasagne.updates.total_norm_constraint(tensor.grad(tensor.cast(d.g_cost + cost, 'float32'), tparams_gen), 5.0)
        d_grads = lasagne.updates.total_norm_constraint(tensor.grad(d.d_cost, d.params), 5.0)

        generator_gan_updates = lasagne.updates.adam(g_grads,
                                                  tparams_gen, learning_rate,
                                                  beta1)

        discriminator_gan_updates = lasagne.updates.adam(d_grads,
                                                      d.params, learning_rate,
                                                      beta2)

        all_updates = generator_gan_updates
        all_updates.update(discriminator_gan_updates)


        train_generator_discriminator = theano.function(inputs = inps + inps_sampled + [discriminator_target],
                                          outputs = {
                                                     'accuracy' : d.accuracy,
                                                     'classification' : d.classification,
                                                     'loss': d.loss,
                                                     'd_cost': d.d_cost,
                                                     'g_cost': d.g_cost
                                                    },
                                          updates = all_updates)

#'g' : tensor.sum(tensor.abs_(tensor.grad(generator_loss, tparams.values()[0]))#'g_loss' : tensor.grad(generator_loss, tparams.values()[0])},

    t_mb = time.time()

    teacher_forcing_cost = 0
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            print "total time in minibatch", time.time() - t_mb
            t_mb = time.time()

            print "experiment id directory", saveto

            log_entry = {'iteration': uidx}

            # pad batch and create mask
            x, x_mask = prepare_data(x, maxlen, n_words)
            if x is None:
                log.log({'minibatch status' : 'Minibatch with zero sample under length'})
                uidx -= 1
                continue

            bern_dist = numpy.random.binomial(1, .5, size=x.shape)
            uniform_sampling = numpy.random.uniform(size = x.flatten().shape[0])

            ud_start = time.time()

            log_entry['x_shape_before_grad'] =  x.shape
            # compute cost, grads and copy grads to shared variables

            print "x shape before grad", x.shape

            if x.shape[1] != batch_size:
                continue

            cost = f_grad_shared(x.astype('int32'),
                                 x_mask.astype('float32'),
                                 bern_dist.astype('float32'),
                                 uniform_sampling.astype('float32'))

            real_hidden_state = get_hidden(x.astype('int32'),
                                           x_mask.astype('float32'),
                                           bern_dist.astype('float32'),
                                           uniform_sampling.astype('float32'))


            #Taking the last hidden state
            real_hidden_state = real_hidden_state[0]
            real_hidden = real_hidden_state[:][maxlen - 1]


            # do the update on parameters
            f_update(lrate)
            ud = time.time() - ud_start
            print "time to do tf update", ud

            log_entry['update_time'] = ud
            log_entry['cost'] = float(cost)
            teacher_forcing_cost = float(cost)
            log_entry['average_source_length'] = \
                                         float(x_mask.sum(0).mean())


            log.log(log_entry)

            print "Number samples processed", n_samples
            try:
                cost_ma = cost_ma * 0.99 + cost * 0.01
            except:
                cost_ma = cost

            print "Cost Moving Average", cost_ma
            print "Training Likelihood Cost", cost

            # check for bad numbers
            if numpy.isnan(cost) or numpy.isinf(cost):
                LOGGER.info("Nan Detected")
                continue;



            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                log.log({'Saving': uidx})
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                save_params(params, saveto + "/" + model_filename)

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0 and n_samples > gan_starting_point:

                t0 = time.time()
                gensample = [];
                count_gen = 0;

                print "batch sampling"

                sample_batch, score_batch, nss_batch = gen_sample_batch(tparams, f_next, model_options, trng=trng,maxlen = maxlen, argmax=False, initial_state = tparams['initial_hidden_state_0'].get_value())

                gensample = sample_batch

                if random.uniform(0,1) < 0.1:
                    print "saving samples"
                    gensample_arr = numpy.asarray(gensample)
                    print "Sample shape", gensample_arr.shape
                    sample_loc = saveto + "/sample_" + str(n_samples) + ".png"
                    print "Saving to", sample_loc
                    plot_image_grid(gensample_arr[:20,:784].reshape((20,1,28,28)), 4, 5, sample_loc)

                sampling_time = time.time() - t0
                log.log({'Sampling_time': sampling_time})
                print "sampling time", sampling_time

                results = prepare_data(gensample, maxlen, n_words)
                genx, genx_mask = results[0], results[1]

                if genx is None:
                    log.log({'Minibatch with zero sample under length ' : maxlen})
                    continue

                log.log({'x_shape': x.shape,
                         'x_mask_shape': x_mask.shape,
                         'genx_shape': genx.shape,
                         'gen_mask_shape': genx_mask.shape})

                if flag_save_tsne and use_gan_objective:
                    generated_hidden_state = get_hidden_sampled(genx.astype('int32'),
                                                                genx_mask.astype('float32'),
                                                                bern_dist.astype('float32'),
                                                                uniform_sampling.astype('float32'))

                    generated_hidden_state = generated_hidden_state[0]
                    generated_hidden = generated_hidden_state[:][maxlen-1]

                if flag_save_tsne and numpy.mod(uidx, 50) == 0:
                    for i in range(maxlen):
                        generated_hidden = generated_hidden_state[:][i]
                        real_hidden = real_hidden_state[:][i]
                        hidden_state_plotted = numpy.concatenate((real_hidden, generated_hidden), axis=0)
                        target_variable = numpy.asarray(([1] * batch_size) + ([0] * batch_size)).astype('int32')
                        digits_proj = TSNE(random_state=RS).fit_transform(hidden_state_plotted)
                        scatter(digits_proj, target_variable)
                        plt.savefig(experiment_id + '_' + str(i) + '.png', dpi=120)

                    file_names = [ j for j in range(maxlen)]
                    file_names = [experiment_id + str(file_names[j]) + '.png' for j in range(maxlen)]
                    images = [Image.open(fn) for fn in file_names]
                    writeGif(save_tsne + '/' + model_name + '_' + str(uidx) + '.GIF',
                             images, duration=0.5, repeat=False)
                    #plt.savefig(save_tsne + '/' + model_name + '_' + str(uidx) + '.png', dpi=120)

                log.log({'no batch norm' : 'no batch norm'})

                t0_gan = time.time()

                if use_gan_objective and x.shape[1] == batch_size and genx.shape[1] == batch_size:
                    target = numpy.asarray(([1] * batch_size) + ([0] * batch_size)).astype('int32')

                    t0 = time.time()
                    log.log({'last_accuracy': last_acc})
                    if False and train_generator_flag and discriminator_accuracy_moving_average > limit_desc_end:
                        train_msg = "Training generator"



                    elif train_generator_flag and (True or discriminator_accuracy_moving_average > limit_desc_start):

                        t0 = time.time()

                        train_msg = "Training discriminator and generator"
                        disc_lr_mult = numpy.asarray(1.0).astype('float32')
                        gen_lr_mult = numpy.asarray(1.0).astype('float32')

                        results_map = train_generator_discriminator(x, x_mask, bern_dist.astype('float32'),
                                                      uniform_sampling.astype('float32'),
                                                      genx, genx_mask,
                                                      bern_dist.astype('float32'),
                                                      uniform_sampling.astype('float32'), target)




                        gen_loss = results_map['g_cost']
                        disc_loss = results_map['d_cost']
                        log.log({'update type' : "discriminator and generator",
                                  'Generator_Loss': gen_loss,
                                  'Discriminator_Loss': disc_loss})

                    else:
                        pass
                        #train_msg = "Just training discriminator"
                        #results_map = train_discriminator(x, x_mask,
                        #                                  bern_dist.astype('float32'),
                        #                                  uniform_sampling.astype('float32'),
                        #                                  genx, genx_mask, bern_dist.astype('float32'),
                        #                                  uniform_sampling.astype('float32'), target)
                        #gen_loss = float(results_map['g_cost'])
                        #disc_loss = float(results_map['d_cost'])
                        #log.log({'Epoch': eidx,
                        #         'Iteration Number': uidx,
                        #         'update type' : "discriminator",
                        #         'Generator_Loss': gen_loss,
                        #         'Discriminator_Loss': disc_loss})

                    desc_accuracy = float(results_map['accuracy'])
                    single_gen_disc_update =  time.time() - t0
                    gen_loss = float(results_map['g_cost'])
                    disc_loss = float(results_map['d_cost'])
                    c = results_map['classification'].flatten()

                    print "classification shape!", results_map['classification'].shape

                    log.log({'single_gen_disc_update': single_gen_disc_update,
                             'Discriminator_Accuracy': desc_accuracy,
                             'Generator_Loss': gen_loss,
                             'Discriminator_Loss': disc_loss,
                             'Epoch': eidx,
                             'Teacher_forcing_cost': teacher_forcing_cost,
                             'Iteration_Number': uidx,
                             'Mean scores (first should be higher than second)' : (c[:batch_size].mean(), c[batch_size:].mean())})

                    print "time for gan objective", time.time() - t0_gan


                    discriminator_accuracy_moving_average = discriminator_accuracy_moving_average * 0.9 + results_map['accuracy'] * 0.1

                    if random.uniform(0,1) < 0.1:
                        print "================================================================================================================="
                        print train_msg
                        print "Discriminator accuracy", results_map['accuracy']
                        print "Discriminator average accuracy", discriminator_accuracy_moving_average
                        print c
                        print results_map['d_cost'], results_map['g_cost']

                    for i in range(0, 32):
                        sentence_print = ""
                        for j in range(0, maxlen):
                            word_num = x[j][i]
                            #if word_num == 0:
                            #    break
                            if word_num in worddicts_r:
                                sentence_print += worddicts_r[word_num] + " "
                            else:
                                sentence_print += "UNK "
                        if random.uniform(0,1) < 0.01:
                            log.log({"Real_Sentence" : sentence_print,
                                "Real_Sentence_Index" : str(i),
                                "Real_Sentence_Accuracy" :str(c[i])})

                    for i in range(32, 64):
                        sentence_print = ""
                        for j in range(0, maxlen):
                            word_num = genx[j][i - 32]
                            #if word_num == 0:
                            #    break
                            if word_num in worddicts_r:
                                sentence_print += worddicts_r[word_num] + " "
                            else:
                                sentence_print += "UNK "
                        if random.uniform(0,1) < 0.01:
                            log.log({"Fake_Sentence" : sentence_print,
                                 "Fake_Sentence_Index" : str(i),
                                 "Fake_Sentence" : str(c[i])})

                    if numpy.isnan(c[:32].mean()) or numpy.isinf(c[:32].mean()) or numpy.isnan(c[32:].mean()) or numpy.isnan(c[32:].mean()):
                        LOGGER.info("Nan Detected with Disc cost!")
                        log.log({'disc status' : "NAN DETECTED"})
                        ipdb.set_trace()
                        continue;

                    log.log({'Mean_pos_scores': c[:32].mean(),
                             'Mean_neg_scores': c[32:].mean()})

                    #print "hidden states joined", results_map['hidden_states'].shape

                    last_acc = results_map['accuracy']

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs, avg_len = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid, maxlen)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                       numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        LOGGER.info("Early Stop!")
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                
                valid_bpc = (valid_err * 1.44)/avg_len

                log.log({"Iteration" : uidx, 'Valid_Err': valid_err,
                    'Avg_Len':avg_len, 'Valid_bpc' : valid_bpc})

                print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                print ""
                print "Iteration", uidx
                print "Validation Error Computed", valid_err
                print "Validation BPC", valid_bpc
                print ""
                print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

            # finish after this many updates
            if uidx >= finish_after:
                log.log({'Finishing_after': uidx})
                estop = True
                break

        log.log({"Samples seen" : n_samples})

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err, avg_len = pred_probs(f_log_probs, prepare_data,
                           model_options, valid, maxlen).mean()

    log.log({'Valid_Err': valid_err})

    params = copy.copy(best_p)
    save_params(params, saveto + model_filename)

    return valid_err


if __name__ == '__main__':
    LOGGER.info('Connecting to worker')
    worker = Worker(control_port=5568)
    LOGGER.info('Retrieving configuration')
    config = worker.send_req('config')
    train(worker, config['model'], config['data'],**merge(config['training'], config['management'], config['multi'],config['model'], config['data']))


