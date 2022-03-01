from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd

def part_of_speech(text):
    # load tagger
    tagger = SequenceTagger.load("flair/pos-english")

    # # make example sentence
    sentence = Sentence(text)

    # predict NER tags
    tagger.predict(sentence)

    # print sentence
    # print(sentence)

    # pos_df = pd.DataFrame(columns=['word','part_speech'])

    # https://github.com/flairNLP/flair/issues/2299
    # print predicted NER spans
    column_names = ["word", "part_speech"]

    # practice_sentence = "Many algorithms have been recently developed for reducing dimensionality by projecting data onto an intrinsic non-linear manifold. Unfortunately, existing algorithms often lose significant precision in this transformation. Manifold Sculpting is a new algorithm that iteratively reduces dimensionality by simulating surface tension in local neighborhoods. We present several experiments that show Manifold Sculpting yields more accurate results than existing algorithms with both generated and natural data-sets. Manifold Sculpting is also able to benefit from both prior dimensionality reduction efforts."

    # using split()
    # to count words in string
    sentence_length = len(text.split())

    pos_df = pd.DataFrame(index=range(sentence_length+6), columns = column_names)


    # print('The following NER tags are found:')
    # iterate over entities and print
    count = 0
    pos1 = []


    for entity in sentence.get_spans('pos'):

        for token in entity:
            # Put each piece into df
            pos1.append([token, token.labels])

            count += 1


    pos_df = pd.DataFrame(pos1, columns=column_names)
    #pos_df['text'] = pos_df['text'].iloc[-1]
    print(pos_df)

    #pos_df[['token','word']] = pos_df["text"].str.split(" ", 1, expand=False)
    pos_df['word'] = pos_df['word'].astype('string')
    pos_df['word'] = pos_df['word'].str.split(' ').str[2]

    pos_df.to_csv('pos.csv')


    return pos_df