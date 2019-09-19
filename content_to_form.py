"""
Concetti: politics, justice, greed, patience, food, vehicle, screw, radiator
"""
import time
from operator import itemgetter

import gensim
import pandas as pd
from nltk import PorterStemmer
from nltk import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.wsd import lesk
from utils.time_it import timeit


def preprocess_sentence(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence_set = set(tokenizer.tokenize(sentence)) - set(stopwords.words('english'))
    return sentence_set


@timeit
def compute_wmd_with_relevant_words(word, definitions):
    """ calculate distance between two sentences using WMD algorithm """
    model = create_model()
    min_dist = 1000
    total_min_dist = 1000
    syns = []
    restricted_synsets = []
    prep_definitions = [d for d in definitions if "opposed" not in d if "without" not in d]
    best_def = ""
    unic_def = " ".join(definitions)
    for defin in prep_definitions:
        for w in preprocess_sentence(defin):
            synsets = wn.synsets(w)
            for s in synsets:
                restricted_synsets += s.hypernyms()
                restricted_synsets += s.hyponyms()
    for ss in set(restricted_synsets):
        distance = model.wmdistance(list(find_relevant_word(unic_def)), list(preprocess_sentence(ss.definition())))
        if distance < total_min_dist:
            total_min_dist = distance
        syns.append((round(total_min_dist, 3), ss))
    best_sense = min(syns, key=itemgetter(0))
    print('synset for word {} is {} because "{}" is the most similar definition of "{}" to "{}"'
          .format(word, best_sense, find_relevant_word(unic_def), word, best_sense[1].definition()))


def compute_wmd(word, definitions):
    """ calculate distance between two sentences using WMD algorithm """
    model = create_model()
    min_dist = 1000
    total_min_dist = 1000
    syns = []
    restricted_synsets = []
    prep_definitions = [d for d in definitions if "opposed" not in d if "without" not in d]
    best_def = ""
    for defin in prep_definitions:
        for w in preprocess_sentence(defin):
            synsets = wn.synsets(w)
            for s in synsets:
                restricted_synsets += s.hypernyms()
                restricted_synsets += s.hyponyms()
    for ss in set(restricted_synsets):
        syns_def = ss.definition().lower().split()
        for d in definitions:
            distance = model.wmdistance(d.lower().split(), syns_def)
            if distance < min_dist:
                min_dist = distance
                best_def = d
        if min_dist < total_min_dist:
            total_min_dist = min_dist
            syns.append((round(total_min_dist, 3), ss))
    best_sense = min(syns, key=itemgetter(0))
    print('synset for word {} is {} because "{}" is the most similar definition of "{}" to "{}"'
          .format(word, best_sense, best_def, word, best_sense[1].definition()))


def find_relevant_word(definition):
    return set(w for w in preprocess_sentence(definition) if
               sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(w), definition)) > 1)


def create_model():
    print('loading model...')
    start = time.perf_counter()
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    end = time.perf_counter()
    print('model loaded in {} seconds'.format(round(end - start)))
    model.init_sims(replace=True)
    return model


def compute_lexical_overlap(word, definition):
    max_overlap = 0
    best_synsets = []
    stemmer = PorterStemmer()
    stemmed_def = [stemmer.stem(w) for w in definition]
    for ss in wn.all_synsets():
        lex_over = lexical_overlap(ss.definition().split(), definition)
        if lex_over >= max_overlap:
            max_overlap = lex_over
            best_synsets.append((ss, max_overlap))
    print("best sysnets for the definition of {} are {}".format(word, sorted(best_synsets, key=itemgetter(1),
                                                                             reverse=True)[:5]))


def lexical_overlap(d1, d2):
    d1 = set(d1)
    d2 = set(d2)
    return len(list(d1 & d2)) / min(len(d1), len(d2))


def run_word_movers_distance():
    compute_wmd('Justice', justice_definitions)
    compute_wmd('Politics', politics_definitions)
    compute_wmd('Greed', greed_definitions)
    compute_wmd('Radiator', radiator_definitions)
    compute_wmd('Patience', patience_definitions)
    compute_wmd('Food', food_definitions)
    compute_wmd('Vehicle', vehicle_definitions)
    compute_wmd('Screw', screw_definitions)


def run_lexical_overlap():
    compute_lexical_overlap('Justice', " ".join(justice_definitions).split())
    compute_lexical_overlap('Politics', " ".join(politics_definitions).split())
    compute_lexical_overlap('Greed', " ".join(greed_definitions).split())
    compute_lexical_overlap('Radiator', " ".join(radiator_definitions).split())
    compute_lexical_overlap('Patience', " ".join(patience_definitions).split())
    compute_lexical_overlap('Food', " ".join(food_definitions).split())
    compute_lexical_overlap('Vehicle', " ".join(vehicle_definitions).split())
    compute_lexical_overlap('Screw', " ".join(screw_definitions).split())


if __name__ == '__main__':
    df_definitions = pd.read_excel("Esperimento content-to-form.xlsx")
    justice_definitions = df_definitions['Justice'].tolist()
    politics_definitions = df_definitions['Politics'].tolist()
    greed_definitions = df_definitions['Greed'].tolist()
    radiator_definitions = df_definitions['Radiator'].tolist()
    patience_definitions = df_definitions['Patience'].tolist()
    food_definitions = df_definitions['Food'].tolist()
    vehicle_definitions = df_definitions['Vehicle'].tolist()
    screw_definitions = df_definitions['Screw'].tolist()

    # run_word_movers_distance()
    # run_lexical_overlap()
