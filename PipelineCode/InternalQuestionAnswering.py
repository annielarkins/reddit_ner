import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from transformers import pipeline
import itertools
from operator import itemgetter

def in_question_answering(text, keywords):
    reader = pipeline("question-answering", model='deepset/roberta-base-squad2')

    # Find phrases in the text
    words_of_interest = keywords[keywords['ner'] != 'NA']
    data = list(words_of_interest.index)
    keyword_ranges = []
    for k, g in itertools.groupby(enumerate(data), lambda ix : ix[0] - ix[1]):
        keyword_ranges.append(list(map(itemgetter(1), g)))

    # Get them out of dataframe
    key_phrases = []
    small_dfs = []
    for kwr in keyword_ranges:
        current_range = words_of_interest.loc[kwr]
        current_type = current_range.iloc[0]['ner']
        current_phrase = ''.join(current_range['words'].replace('', ' '))
        if current_phrase not in key_phrases:
            key_phrases.append(current_phrase)
        if current_type == "description":
            question = "What does " + current_phrase + " mean?"
        elif current_type == "action":
            question = "What does it mean to " + current_phrase + "?"
        else:
            question = "What is " + current_phrase + "?"

        outputs = reader(question=question, context=text)
        results = pd.DataFrame.from_records([outputs])
        if results['score'][0] > .03:
            # print(question)
            row_to_add = pd.DataFrame.from_records([outputs])[['score', 'answer']]
            row_to_add['word'] = current_phrase
            small_dfs.append(row_to_add)
    return pd.concat(small_dfs, ignore_index=True)
