from joblib import load
import pandas as pd
import glob
from spellchecker import SpellChecker
import spacy
import string
import re
from sklearn.base import TransformerMixin
from spacy.lang.es.stop_words import STOP_WORDS

# Installing dependencies
"""Install the libraries and other tools which will be used on the project
    from requirements_min.txt file running the following command:
    pip install -r requirements_min.txt
"""
# Run this commands to install missing dependencies and update the packaging tools pip, setuptools and wheel
# pip install -U pip setuptools wheel
# python -m spacy download es_core_news_sm


# utils to clean the text

spell = SpellChecker(language="es")
nlp = spacy.load("es_core_news_sm")
punctuations = string.punctuation


def remove_urls(word):
    parsed_text = re.sub(r"\S*https?:\S*", "", word, flags=re.MULTILINE)
    return parsed_text


def spellcheck_correct(text):
    corrected_text = spell.correction(text)
    if corrected_text is None:
        corrected_text = text
    return corrected_text


def spacy_tokenizer(text):
    # Create tokens list and correct the spelling of the text
    corrected_text = spellcheck_correct(text)
    tokens = nlp(corrected_text)

    # Lemmatize and lowercase each token
    tokens = [word.lemma_.lower().strip() if word.lemma_ !=
              "PROPN" else word.lower_ for word in tokens]

    # Remove stopwords from token list
    tokens = [
        word for word in tokens if word not in STOP_WORDS and word not in punctuations]

    # Remove links from token list
    tokens = [remove_urls(word) for word in tokens]

    # return the tokens list preprocessed
    return tokens


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text


def clean_text(text):
    return text.strip().lower()


# Load the trained model
logistic_regression_model = load('logistic_regression_model.joblib')

"""
    Comment the following code if you want to classify the tweets from the API
"""
# Perform the prediction
files_to_classify = './testing'
files_path_list = glob.glob(f'{files_to_classify}/*.csv')

multiple_df = [pd.read_csv(file_path, encoding='utf-8-sig')
               for file_path in files_path_list]
dataframe_to_classify = pd.concat(multiple_df)

# Clean the dataset to delete None values, urls, etc.
clean_df_to_classify = [" ".join(spacy_tokenizer(text))
                        for text in dataframe_to_classify['text']]

predicted_labels = logistic_regression_model.predict(clean_df_to_classify)
classified_df = pd.DataFrame(
    {'text': dataframe_to_classify['text'], 'label': predicted_labels}
)
classified_df.to_csv('./classified_dataset/classified_data.csv', index=False)

"""
    Uncomment the following code to classify the tweets from the API and save the results in a new file
"""

# import tweepy as tw

# # Access Tokens
# api_key = "MY_API_KEY"
# api_secret = "MY_API_SECRET"

# # Authentication
# auth = tw.OAuthHandler(api_key, api_secret)
# api = tw.API(auth, wait_on_rate_limit=True)

# # Search query parameters
# search_query = 'covid -filter:retweets'

# # Get tweets from the API
# tweets = tw.Cursor(api.search, q=search_query, lang="es", since="2020-03-21", until="2020-03-26").items(6000)

# # Store the responses
# tweets_list = []
# for tweet in tweets:
#   tweets_list.append(tweet)

# # Verify the length of the list
# print("Total tweets fetched: ", len(tweets_list))

# # Initializing the dataframe
# tweets_df = pd.DataFrame()

# # Populate the dataframe
# for tweet in tweets_list:
#   hashtags = []
#   try:
#     for hashtag in tweet.entities["hashtags"]:
#       hashtags.append(hashtag["text"])
#     text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
#   except:
#     pass
#   tweets_df = tweets_df.append(pd.DataFrame({
#       'user_name': tweet.user.name,
#       'user_location': tweet.user.location,
#       'user_description': tweet.user.description,
#       'user_verified': tweet.user.verified,
#       'date': tweet.created_at,
#       'text': text,
#       'hashtags': [hashtags if hashtags else None],
#       'source': tweet.source
#   }))
#   tweets_df = tweets_df.reset_index(drop=True)

# # Show the dataframe
# tweets_df.head()

# # Clean the dataframe to perform classification
# clean_tweets_df = [" ".join(spacy_tokenizer(text)) for text in tweets_df['text']]

# # Predict using the model and the clean dataframe
# predicted_labels = logistic_regression_model.predict(clean_tweets_df)

# # Save the classified dataset in a new file
# classified_df = pd.DataFrame(
#     {'text': tweets_df['text'], 'label': predicted_labels}
# )
# classified_df.to_csv("./classified_dataset/classified_data.csv", index=False)
