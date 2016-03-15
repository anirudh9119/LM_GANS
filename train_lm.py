from lm_worker import train
#from lm_gen import train

#reload true for lm_gen and
#relaod False for lm_worker


def main(job_id, params):
    print params
    validerr = train(
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_word=params['dim_word'][0],
        dim=params['dim'][0],
        n_words=params['n-words'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        maxlen=30,
        batch_size=32,
        valid_batch_size=32,
        validFreq=1000,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=10,
        dataset='wiki_small.txt.gz',
        #dataset='/data/lisatmp3/chokyun/wikipedia/extracted/wiki.tok.txt.gz',
        valid_dataset='/data/lisatmp4/anirudhg/newstest2011.en.tok',
        dictionary='/data/lisatmp4/anirudhg/wiki.tok.txt.gz.pkl',
        use_dropout=params['use-dropout'][0],
        train_generator_flag=params['train_generator_flag'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['./models/model_lm_nogan.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True],
        'train_generator_flag' : [False]})


