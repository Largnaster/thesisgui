from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import re
import string
import time

import pandas as pd
import spacy
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from spacy.lang.es.stop_words import STOP_WORDS
from spellchecker import SpellChecker
from tqdm import tqdm

# Installing dependencies
"""Install the libraries and other tools which will be used on the project
    from requirements.txt file running the following command:
    pip install -r requirements.txt
"""
# Run this commands to install missing dependencies and update the packaging tools pip, setuptools and wheel
# pip install -U pip setuptools wheel
# python -m spacy download es_core_news_sm

start_time = time.perf_counter()

# List of files
dataframe_to_classify = pd.read_csv(
    './pre_classified/pre_classified_dataset.csv', encoding='utf-8-sig')


spell = SpellChecker(language="es")

# Split the train test
train_df, test_df = train_test_split(dataframe_to_classify, test_size=0.3)
test_df, validation_df = train_test_split(test_df, test_size=0.3)

# Clean tweets into tokens to be digested by different models
""" In this process is necessary to delete all the unwanted characters,
punctuation signs, emojis, urls, hashtags, etc. in order to have useful
information
"""
nlp = spacy.load("es_core_news_sm")

# Add spell check


def spellcheck_correct(text):
    corrected_text = spell.correction(text)
    if corrected_text is None:
        corrected_text = text
    return corrected_text


# Create list of punctuation marks
punctuations = string.punctuation

# Function to remove links


def remove_urls(word):
    parsed_text = re.sub(r"\S*https?:\S*", "", word, flags=re.MULTILINE)
    return parsed_text

# Function to retrieve the tokens from sentences


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


# Clean the data to process the text data
print("Cleaning the data...")
train_df['claim'] = tqdm(train_df['claim'].apply(lambda x: spacy_tokenizer(x)))

# Group tokens into text representation
print("Grouping tokens into text representation...")
train_df['claim'] = tqdm(train_df['claim'].apply(lambda x: " ".join(x)))

# Custom transformer class using spacy


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


# Convert the text to a numeric representation
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))

print("Training models for classification...")
# Multinominal Naive Bayes classifier
classifier_nb = MultinomialNB()
pipe_nb = Pipeline([("cleaner", predictors()), ('vectorizer',
                                                bow_vector), ('classifier', classifier_nb)])

tqdm(pipe_nb.fit(train_df['claim'], train_df['label']))
predicted_nb = pipe_nb.predict(test_df['claim'])

print(classification_report(test_df['label'],
      predicted_nb))
print("Classification with Naive Bayes model finished with {} seconds".format(
      time.perf_counter() - start_time)
      )


# Logistic Regression classifier
classifier_log = LogisticRegression()
pipe_log = Pipeline([("cleaner", predictors()), ('vectorizer',
                    bow_vector), ('classifier', classifier_log)])

tqdm(pipe_log.fit(train_df['claim'], train_df['label']))
predicted_log = pipe_log.predict(test_df['claim'])

print(classification_report(test_df['label'], predicted_log))
print("Classification with Logistic Regression model finished with {} seconds".format(
      time.perf_counter() - start_time))

# SVM classifier
classifier_svm = SVC()
pipe_svm = Pipeline([("cleaner", predictors()), ('vectorizer',
                    bow_vector), ('classifier', classifier_svm)])

tqdm(pipe_svm.fit(train_df['claim'], train_df['label']))
predicted_svm = pipe_svm.predict(test_df['claim'])

print(classification_report(test_df['label'], predicted_svm))
print("Classification with SVM model finished with {} seconds".format(
      time.perf_counter() - start_time))

# Random Forest classifier
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
pipe_rf = Pipeline([("cleaner", predictors()), ('vectorizer',
                   bow_vector), ('classifier', classifier_rf)])

tqdm(pipe_rf.fit(train_df['claim'], train_df['label']))
predicted_rf = pipe_rf.predict(test_df['claim'])

print(classification_report(test_df['label'], predicted_rf))
print("Classification with Random Forest model finished with {} seconds".format(
    time.perf_counter() - start_time))


print("DONE! execution time: ", time.perf_counter() - start_time, " seconds")

# Save the trained model using joblib
model_file_name = 'logistic_regression_model.joblib'
dump(pipe_log, model_file_name)
print("Model saved as ", model_file_name)
