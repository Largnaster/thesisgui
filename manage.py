#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import spacy
from spellchecker import SpellChecker
from spacy.lang.es.stop_words import STOP_WORDS
from joblib import load
import re
import string
import os
from pathlib import Path
import sys
from sklearn.base import TransformerMixin


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [self.clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    def clean_text(self, text):
        return text.strip().lower()


# utils to clean the text
spell = SpellChecker(language="es")
nlp = spacy.load("es_core_news_sm")
punctuations = string.punctuation

# Load the trained model
base_path = Path(__file__).resolve().parent
model_path = os.path.join(
    base_path, 'classification', 'models', 'logistic_regression_model.joblib')


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

# Basic function to clean the text


def get_model():
    logistic_regression_model = load(model_path)
    return logistic_regression_model


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'thesisgui.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
