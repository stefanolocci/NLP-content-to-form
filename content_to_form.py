"""
Concetti: politics, justice, greed, patience, food, vehicle, screw, radiator
"""
import itertools
import time
from operator import itemgetter

import gensim
import pandas as pd
from nltk import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer

from utils.time_it import timeit


def preprocess_sentence(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence_set = set(tokenizer.tokenize(sentence)) - set(stopwords.words('english'))
    return sentence_set


@timeit
def compute_wmd(word, definitions):
    """ calculate distance between two sentences using WMD algorithm """
    min_dist = 9999999999999
    min_syns_def_dist = 999999999
    syns = []
    restricted_synsets = []
    prep_definitions = [d for d in definitions if "opposed" not in d if "without" not in d]
    for defin in prep_definitions:
        for w in preprocess_sentence(defin):
            synsets = wn.synsets(w)
            for s in synsets:
                restricted_synsets += s.hypernyms()
                restricted_synsets += s.hyponyms()
                # restricted_synsets += itertools.chain(*s.hypernym_paths())
    for ss in set(restricted_synsets):
        for d in definitions:
            distance = model.wmdistance(d.split(), ss.definition().lower().split())
            if distance < min_dist:
                min_dist = distance
        if min_dist < min_syns_def_dist:
            min_syns_def_dist = min_dist
            syns.append((round(min_syns_def_dist, 3), ss))
    print('synsets for word {} are {}'.format(word, sorted(syns, key=itemgetter(0))))


def find_relevant_word(definition):
    return set(w for w in preprocess_sentence(definition) if
               sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(w), definition)) > 1)


df_definitions = pd.read_excel("Esperimento content-to-form.xlsx")
justice_definitions = df_definitions['Justice'].tolist()
politics_definitions = df_definitions['Politics'].tolist()
greed_definitions = df_definitions['Greed'].tolist()
radiator_definitions = df_definitions['Radiator'].tolist()
patience_definitions = df_definitions['Patience'].tolist()
food_definitions = df_definitions['Food'].tolist()
vehicle_definitions = df_definitions['Vehicle'].tolist()
screw_definitions = df_definitions['Screw'].tolist()

print('loading model...')
start = time.perf_counter()
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
end = time.perf_counter()
print('model loaded in {} seconds'.format(round(end - start)))
# model.init_sims(replace=True)

exclusion_words = ['opposed', 'without', 'opposite']

compute_wmd('Justice', justice_definitions)
compute_wmd('Politics', politics_definitions)
compute_wmd('Greed', greed_definitions)
compute_wmd('Radiator', radiator_definitions)
compute_wmd('Patience', patience_definitions)
compute_wmd('Food', food_definitions)
compute_wmd('Vehicle', vehicle_definitions)
compute_wmd('Screw', screw_definitions)

