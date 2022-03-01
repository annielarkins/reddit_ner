from nltk.corpus import words
import nltk
import enchant
from nltk.stem import WordNetLemmatizer

nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

# function for stemming
def get_lemm(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    # print(wordnet_lemmatizer.lemmatize('developed'))
    # print (wordnet_lemmatizer.lemmatize("geese"))
    text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in text.split()])
    # print([wordnet_lemmatizer.lemmatize(word) for word in text.split()])
    return text

def dictionary_comparison(text):
    abstract1 = get_lemm(text)

    abstract_lower = abstract1.lower()

    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    
    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in abstract_lower:
        if ele in punc:
            abstract_lower = abstract_lower.replace(ele, "")

    res = abstract_lower.split() 

    # printing result 
    #for i in res:
    #if not (i in words.words()):
        #print(i)

    keywords = []
    for i in res:
        d = enchant.Dict("en_US")
        if (not (d.check(i))):
            keywords.append(i)
            # print(i)
    return keywords

