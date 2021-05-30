from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import pickle
import pkg_resources
import pathlib

import re, string, random, os

#remove unwanted noise from tweet and lemmetize it
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

#classifies the text as either positive for depression or negative for depression. 
def classify(text, mode="string", switchpoint=0.95, model_id=31):
    classifier = get_classifier(model_id)

    custom_tokens = remove_noise(word_tokenize(text))

    dist = classifier.prob_classify(dict([token, True] for token in custom_tokens))

    #string mode returns results as string
    if mode == "string":
        for label in dist.samples():
            if label == "Positive" and dist.prob(label) > switchpoint:
                return "Positive"
        return "Negative"

    #int mode returns results as binary ints
    elif mode =="int":
        for label in dist.samples():
            if label == "Positive" and dist.prob(label) > switchpoint:
                return 1
        return 0

    #probabilities mode returns result as a list of float probabilities
    elif mode=="probabilities":
        return_list = []
        for label in dist.samples():
            return_list.append((label, dist.prob(label)))
        return return_list

    else:
        raise Exception("Mode must be [string/int/probabilites]")
            

    


#loads the pickle file into a Naive Bayes Classifier
def get_classifier(id):
    
    path = pathlib.Path(str(__file__))
    path = path.parent
    path = os.path.join(path, os.path.join('data', f'{id}.pickle'))
    f = open(path, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier