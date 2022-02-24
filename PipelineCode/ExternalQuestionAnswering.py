import requests
from IPython.display import HTML
import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from transformers import pipeline
import itertools
from operator import itemgetter

subscription_key = "4806df7ed1b0468ba9e38399daad530a"
assert subscription_key

def query_bing(search_term, domain=''):
    st = search_term + " " + domain
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": st, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    # rows = "\n".join(["""<tr>
    #                      <td><a href=\"{0}\">{1}</a></td>
    #                      <td>{2}</td>
    #                    </tr>""".format(v["url"], v["name"], v["snippet"])
    #                 for v in search_results["webPages"]["value"]])
    potential_defs = [ x['snippet'] for x in search_results['webPages']['value'][0:3]]
    answer_df = pd.DataFrame([potential_defs], columns = ['SearchResult1', 'SearchResult2', 'SearchResult3'])
    answer_df['word'] = search_term
    return answer_df

def ex_question_answering(text, keywords):
    # Find phrases in the text
    words_of_interest = keywords[keywords['ner'] != 'NA']
    data = list(words_of_interest.index)
    keyword_ranges = []
    for k, g in itertools.groupby(enumerate(data), lambda ix : ix[0] - ix[1]):
        keyword_ranges.append(list(map(itemgetter(1), g)))

    key_phrases = []
    small_dfs = []
    # Get them out of dataframe
    for kwr in keyword_ranges:
        current_range = words_of_interest.loc[kwr]
        current_type = current_range.iloc[0]['ner']
        current_phrase = ''.join(current_range['words'].replace('', ' '))
        if current_phrase not in key_phrases:
            small_df = query_bing(current_phrase)
            small_dfs.append(small_df)
            key_phrases.append(current_phrase)
    return pd.concat(small_dfs, ignore_index=True)
                    

