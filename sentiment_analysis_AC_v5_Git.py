
########################################
### Sentiment Analysis Model ###
########################################
print ("Sentiment Analysis Model")
print ("Importing the libraries")
# DataFrame
import pandas as pd
# Matplot
# import matplotlib.pyplot as plt
#%matplotlib inline
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
# Word2vec
import gensim
# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools



# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')

print("Setting the variable")
# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLEANING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
# W2V_EPOCH = 32
W2V_EPOCH = 5
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
# EPOCHS = 4
EPOCHS = 2
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"



print("Reading the data")
### Read the dataset
dataset_path = "/Users/adrianocarneiro/Google Drive/_20170918_Graduacao_BYU/_202001_Winter_2020/CS_450_ML/Projects/W2020_CS450_ML/Final_Project/Twitter_Sentimental_Analysis_Git_Project_v3/data/training.1600000.processed.noemoticon.csv"
df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
### --- ###


### USING ONLY PART OF THE DATA TO TEST PURPOSE
# df, _ = train_test_split(df1, test_size=0.3, train_size=0.4, shuffle = True)
### --- ###

print("Replacing target number for text labels")
### Replacing number in the target column to name labels
# A dictionary with the key number and the value name label
decode_label_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
# Method that return a name label based on the code
def decode_sentiment(code):
    return decode_label_map[int(code)]
### --- ###

print("Pre-processing the data")
### Pre-processing the data!
# The process of converting data to something a computer can understand is referred to as pre-processing.
# One of the major forms of pre-processing is to filter out useless data.
# stopwords: In natural language processing, useless words (data), are referred to as stop words(such as “the”, “a”, “an”, “in”)
stop_words = stopwords.words("english")
# SnowballStemmer:
# keep root and stem words - ‘touched’ the stem is ‘touch’; in the form ‘wheelchairs’ the stem is ‘wheelchair’
# re- + friger + -ate + -tor = prefix + root + 2 suffixes.
stemmer = SnowballStemmer("english")
# Cleaning the text
def preprocess(text, stem=False):
     # Remove link,user and special characters
     text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
     cleaned_words = []
     for word in text.split(): # split by space getting each word in the text
         if word not in stop_words:
             if stem:
                 cleaned_words.append(stemmer.stem(word))
             else:
                 cleaned_words.append(word)
     return " ".join(cleaned_words)
# getting each text in dataset and applying the preprocess method
df.text = df.text.apply(lambda x: preprocess(x))
### --- ###

print("Splitting the data")
# splitting dataset into test and training
#df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
# shuffle: whether or not to shuffle the data before splitting
df_train, df_test = train_test_split(df, test_size=0.3, train_size=0.7, shuffle = True)

### Gensim - Word2Vec
# Word2Vec: is that two words sharing similar contexts also share a similar meaning and consequently a similar vector
# For instance: "dog", "puppy" and "pup" are often used in similar situations
# used to find out the relations between words in a dataset, compute the similarity between them
# Use the vector representation of those words as input for other applications such as text classification or clustering

# Creating a list of arrays. Each row text will be splitted by white space
# [row_1: ["I", "like", "Rexburg"] row_2: ["This", "app", "works", "well"]]
documents = [_text.split() for _text in df_train.text]

print("Setting up and training the work to vector model")
#  Setting up the word2vec model
w2v_model = gensim.models.word2vec.Word2Vec( size=W2V_SIZE,
                                             window=W2V_WINDOW,
                                             min_count=W2V_MIN_COUNT,
                                             workers=8)

# Building a vocabulary based on the array of words(document) and the parameter set previously
w2v_model.build_vocab(documents)
 # Getting the words created

# words = w2v_model.wv.vocab.keys()
# vocab_size = len(words)
# print("Vocab size", vocab_size)

# Training the model
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

# Get similar words based on the trained model
w2v_model.most_similar("love")
### --- ###

print("Changing train and test attributes(tweet content) from text to number | Preparing to run train and test the Neural network model | Tokenizer")
### Changing the Tweet texts(x_train, x_test) unto a number sequences make it ease to train the model
# Tokenizer: This class allows to vectorize a text corpus, by turning each text into either a sequence of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)
vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)
# Train and test data used to test the neural network model
x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
### --- ###

print("Changing train and test labels(positive, negative) from text to number | Preparing to run train and test the Neural network model | LabelEncoder")
### Changing the Tweet labels(y_train, y_test) unto a number sequences make it ease to train the model
encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())
y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
### --- ###

print("Setting up training and testing a neural network | Tensorflow-Keras")
### Commented because I am using the saved model
"""
### Neural Network - Setting up, Training, Testing ###
# Setting up the neural network
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

#print(embedding_matrix.shape)

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
               optimizer="adam",
               metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
               EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

# Training the neural network
history = model.fit(x_train, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=EPOCHS,
                     validation_split=0.1,
                     verbose=1,
                     callbacks=callbacks)

# Testing the neural network
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print("ACCURACY:",score[1])
print("LOSS:",score[0])
### --- ###
"""

print("Saving / Loading the model")
### Saving / Loading the model ###
from keras.models import load_model
# model.save('twitter_BOT_V3')
#del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
model = load_model('/Users/adrianocarneiro/Google Drive/_20170918_Graduacao_BYU/_202001_Winter_2020/CS_450_ML/Projects/W2020_CS450_ML/Final_Project/Twitter_Sentimental_Analysis_Git_Project_v3/twitter_BOT_V3')
### --- ###

# Changing positiveness score to a label score
def decode_predicted_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

# Using the model to make prediction | Returning the score and label
def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_predicted_sentiment(score, include_neutral=include_neutral)
    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}

# Using the model to make prediction | Returning the positiveness score
def predict_positiveness_score(text):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    return score

# predict("I love the music")


########################################
### Twitter Connection API ###
########################################
print ("Twitter Connection API")
print ("Importing the libraries")
# Tweepy to access Twitter API
from tweepy import OAuthHandler, Stream, StreamListener
import tweepy

# TextBlob used to classify text as positive negative -1(negative) to 1(positive)
from textblob import TextBlob
from geopy.geocoders import Nominatim

# Flask to access the make methods accessble as API
from flask import Flask
from flask import Flask, request
from flask import jsonify
from flask import Response

# DataFrame
import pandas as pd

print("Setting the variable")
# Credentials to access the Twitter API 
consumer_key = '5VRb3V2d31stqgAeEMZjoGozC'
consumer_secret = 'HPlOOacZHCEc80TREwlHPqHqDtYG9QfmsDddkiNzJpYNrC4l00'
access_token = '322916754-kkzhIhKkq6Fne7RnEcxHsFMtanXDesU99olhuKwu'
access_token_secret = 'k5juTBfnMRgFm2ZuuDpJQQ65xntlBFl6jwm2iGJrcibVN'

app = Flask(__name__)

# Using Twitter API to access tweets based on the searched term and number of tweets
def GetTweets(searchedTerm, numberTweets):
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        query = searchedTerm
        tweet_returned_list = [status for status in tweepy.Cursor(api.search, q=query).items(numberTweets)]
        
        return tweet_returned_list
    except Exception as e: 
            # print(e)
            return 'Unfortunately we got an error in our Sentiment Analysis API {0}'.format(e)

# API Welcome message
@app.route('/')
def Welcome():
    return 'Welcome to the Twitter sentiment analysis API'

# API available method to get the tweets and its geolocation
@app.route('/GetTweetsSentiment/<string:searchedTerm>/<int:numberTweets>', methods=['GET'])
def GetTweetsSentiment(searchedTerm, numberTweets):
    try:
        sentiment_analysis_dataset = pd.DataFrame()
        tweet_returned_list = GetTweets(searchedTerm, numberTweets)
        geolocator = Nominatim()

        # df.text = df.text.apply(lambda x: preprocess(x))
        for tweet_returned in tweet_returned_list:
            if (len(tweet_returned.user.location) != 0):
                # Getting the geolocation based on the users location field
                tweet_location = geolocator.geocode(tweet_returned.user.location)
                if (tweet_location is not None and
                        tweet_location.latitude is not None and
                        tweet_location.longitude is not None
                    ):
                    tweet_latitude = tweet_location.latitude
                    tweet_longitude = tweet_location.longitude
                    
                    # tweet_returned_cleaned = PreprocessingText(tweet_returned.text)
                    # tweet_sentiment_analysis = TextBlob(tweet_returned_cleaned)
                    # tweet_positiveness = tweet_sentiment_analysis.sentiment.polarity
                    
                    # Call Sentiment Analysis Model class to predict the tweets positiveness score
                    tweet_positiveness = predict_positiveness_score(tweet_returned.text)
                    df = pd.DataFrame({ "positiveness_score": [tweet_positiveness[0]], 
                                        "latitude": [tweet_latitude],
                                        "longitude": [tweet_longitude]#,
                                        #"tweet":[tweet_returned.text]
                                      })
                    sentiment_analysis_dataset = sentiment_analysis_dataset.append(df)
        # return sentiment_analysis_dataset
        return Response(sentiment_analysis_dataset.to_json(orient="records"), mimetype='application/json')

    except Exception as e: 
        return 'Unfortunately we got an error in our Twitter Connection API {0}'.format(e)


if __name__ == '__main__':
    print("API is Running...")
    app.run(debug=True, use_reloader=False, threaded=False)

    # r = GetTweetsSentiment("Trump", 10)

    # x = 1123


