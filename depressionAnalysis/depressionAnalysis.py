from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import pickle
import pkg_resources
import pathlib
import twint
import json
import csv
import pandas as pd


import re, string, random, os

#https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/

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
def classify(text, mode="string", switchpoint=0.95, model_id=96, classifier=None):

    switchpoint = float(switchpoint)
    model_id = int(model_id)
    if switchpoint < 0 or switchpoint > 1:
        raise Exception("Switchpoint must be between 0 and 1")
    if model_id < 0 or model_id > 96:
        raise Exception("Model id must be between 0 and 96")

    if classifier == None:
        classifier = get_classifier(model_id)

    custom_tokens = remove_noise(word_tokenize(text))

    dist = classifier.prob_classify(dict([token, True] for token in custom_tokens))

    positive_acc = dist.prob("Positive")
    negative_acc = dist.prob("Negative")

    
    #string mode returns results as string
    if mode == "string":
        if positive_acc > switchpoint and positive_acc > negative_acc:
            return "Positive"
        else:
            return "Negative"

    #int mode returns results as binary ints
    elif mode =="int":
        if positive_acc > switchpoint and positive_acc > negative_acc:
            return 1
        else:
            return 0

    #probabilities mode returns result as a list of float probabilities
    elif mode=="probabilities":
        return_list = []
        for label in dist.samples():
            return_list.append((label, dist.prob(label)))
        return return_list

    else:
        raise Exception("Mode must be [string/int/probabilities]")
            


#loads the pickle file into a Naive Bayes Classifier
def get_classifier(id):
    
    
    id = int(id)
    if id < 0 or id > 96:
        raise Exception("Model id must be between 0 and 96")

    path = pathlib.Path(str(__file__))
    path = path.parent
    path = os.path.join(path, os.path.join('data', f'{id}.pickle'))
    f = open(path, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


#gets a list containing the text of a users twitter posts
def get_all_posts_for_user(user):
    return_list = []
        
    #configuration
    config = twint.Config()
    config.Username = user
    config.Lang = "en"
    config.Hide_output = True
    config.Store_object = True

    #run search
    twint.run.Search(config)
    tweets = twint.output.tweets_list

    #get text of tweets
    for tweet in tweets:
        return_list.append(tweet.tweet)

    return return_list

#this function takes a user's twitter handle and uses the classifier to compute whether or not they are depressed, and generate data about that
def analyse_user(user, model_id=96, posts=None, switchpoint=0.8, classification_switchpoint=0.96, save_as=None, filename=None, encoding="utf-8"):
    if posts == None:
        posts = get_all_posts_for_user(user)

    
    total_positive = 0
    total_negative = 0
    average_positive = 0
    average_negative = 0

    #loads the classifier from the pickle file in the pip package
    classifier = get_classifier(model_id)

    #prepares a dictionary to hold all the data collected about the user
    return_dict = {
        "model_id":model_id,
        "username":user,
        "total_posts":len(posts),
        "total_positive":total_positive,
        "total_negative":total_negative,
        "percent_positive":0,
        "percent_negative":0,
        "average_positive_likelihood":average_positive,
        "average_negative_likelihood":average_negative,
        "posts":[],
        "depressed":False,
        "switchpoint":switchpoint,
        "classification_switchpoint":classification_switchpoint

    }

    #goes through each of the user's posts
    for post in posts:

        #gets both a binary classification and a probability classification for each post
        classification = classify(post, classifier=classifier, mode="int", switchpoint=classification_switchpoint)
        probability_classification = classify(post, classifier=classifier, mode="probabilities")
        
        
        if classification == 1:
            total_positive += 1
        else:
            total_negative += 1

        average_positive += probability_classification[1][1]
        average_negative += probability_classification[0][1]

        #adds the data about each post to the posts list in the dictionary
        return_dict["posts"].append({
            "text":post,
            "classification":classification,
            "positive_likelihood":probability_classification[1][1],
            "negative_likelihood":probability_classification[0][1]
        })
    
    #updates dictionary with the collected data
    return_dict["total_positive"] = total_positive
    return_dict["total_negative"] = total_negative
    return_dict["percent_positive"] = float(total_positive/len(posts))
    return_dict["percent_negative"] = float(total_negative/len(posts))
    return_dict["average_positive_likelihood"] = float(average_positive/len(posts))
    return_dict["average_negative_likelihood"] = float(average_negative/len(posts))

    #decides if user is depressed or not based on their average positive likelihood, which is the most accurate way.
    return_dict["depressed"] = (return_dict["average_positive_likelihood"] > switchpoint)
    
    #saves collected data in various data formats for further analysis
    if save_as != None:

        if filename == None:
            filename = f"{user}_depression_statistics"
        
        if save_as == "json":
            with open(f"{filename}.json", "w") as outfile: 
                json.dump(return_dict, outfile, indent=4)

        elif save_as == "csv":
            outfile = open(f"{filename}.csv", "w", encoding=encoding)

            writer = csv.writer(outfile)
            for key, value in return_dict.items():
                if key != "posts":
                    writer.writerow([key, value])

            outfile.close()

        elif save_as == "xlsx":
            #save base stats to excel file
            
            
            df = pd.json_normalize(return_dict)
            df.to_excel(f'{filename}.xlsx', sheet_name=user, index=False)

            posts_data = return_dict["posts"]
            df = pd.json_normalize(posts_data)
            df.to_excel(f'{filename}_posts.xlsx', sheet_name=user, index=False)

        else:
            raise Exception("save_as must be [json/csv/xlsx]")

    #returns data dictionary
    return return_dict
