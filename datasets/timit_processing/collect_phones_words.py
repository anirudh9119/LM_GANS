"""
Collect phones and words from raw TIMIT dataset into dictionary (where keys are utterance names specified in KALDI format).
Map phones and words to integers, and create a dictionary.
"""
import os
import os.path
import pickle

def collect_phones_words(which_set):
    unaligned_phones = {} #dictionary with utterance name as key, and phone list as value
    unaligned_words = {} #dictionary with utterance name as key, and word list as value
    
    PATH = '/data/lisa/data/timit/raw/TIMIT/' + which_set.upper()
    DR_dirs = [d for d in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, d))]

    #Traverse all directories in PATH
    for DR_dir in DR_dirs:
        cur_path = os.path.join(PATH, DR_dir)
        # The directory name is the first identifier of utterance name e.g. FAEM0_*
        first_ids = [d for d in os.listdir(cur_path) if os.path.isdir(os.path.join(cur_path, d))]
        for first_id in first_ids:
            cur_path_2 = os.path.join(cur_path, first_id)
            #File name is second part of utterance name e.g. *_SI1392 means there exist a filename SI1392.PHN in the current directory
            phones_files = [d for d in os.listdir(cur_path_2) if d.endswith('.PHN')]
            
            for phone_file in phones_files:
                utterance_name = first_id + '_' + phone_file.replace('.PHN', '')
                #Need to extract phones only from file (ignore alignment)
                phones = []            
                with open(os.path.join(cur_path_2, phone_file)) as f:
                    for line in f:
                        phones.append(line.replace('\n', '').split(' ')[-1]) #Remove newline and select phone in last column
                unaligned_phones[utterance_name] = phones
                
                #Extract words only from file (ignore alignment)
                words = []
                with open(os.path.join(cur_path_2, phone_file.replace('.PHN', '.WRD'))) as f:
                    for line in f:
                        words.append(line.replace('\n', '').split(' ')[-1]) #Remove newline and select word in last column
                unaligned_words[utterance_name] = words
    return unaligned_phones, unaligned_words

test_phones, test_words = collect_phones_words('test') 
train_phones, train_words = collect_phones_words('train')

#Map phones to integers, and create phone dictionary
phone_symbols = set(reduce(lambda x, y: x + y, [x for x in train_phones.itervalues()] + [x for x in test_phones.itervalues()])) 
phone_mapper = dict([(key, value) for value, key in enumerate(list(phone_symbols), start=2)])
phone_mapper['<START>'] = 0
phone_mapper['<END>'] = 1

train_mapped_phones = {}
for key, value in train_phones.items():
    train_mapped_phones[key] = [0] + [phone_mapper[x] for x in value] + [1] #Map phones and add start and end symbols

test_mapped_phones = {}
for key, value in test_phones.items():
    test_mapped_phones[key] = [0] + [phone_mapper[x] for x in value] + [1]

#Serialize phones and its dictionary
pickle.dump(train_mapped_phones, open('train_phones.pkl', 'wb'))
pickle.dump(test_mapped_phones, open('test_phones.pkl', 'wb'))
pickle.dump(phone_mapper, open('phones_dict.pkl', 'wb'))

#Map words to integers, and create word dictionary
word_symbols = set(reduce(lambda x, y: x+y, [x for x in train_words.itervalues()] + [x for x in test_words.itervalues()])) 
word_mapper = dict([(key, value) for value, key in enumerate(list(word_symbols), start=2)])
word_mapper['<START>'] = 0
word_mapper['<END>'] = 1

train_mapped_words = {}
for key, value in train_words.items():
    train_mapped_words[key] = [0] + [word_mapper[x] for x in value] + [1] #Map words and add start and end sequences

test_mapped_words = {}
for key, value in test_words.items():
    test_mapped_words[key] = [0] + [word_mapper[x] for x in value] + [1]

#Serialize words and its dictionary
pickle.dump(train_mapped_words, open('train_words.pkl', 'wb'))
pickle.dump(test_mapped_words, open('test_words.pkl', 'wb'))
pickle.dump(word_mapper, open('words_dict.pkl', 'wb'))


