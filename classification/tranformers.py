from sklearn.base import TransformerMixin
import spacy
import string
import re
from joblib import load
from django.conf import settings
from spacy.lang.es.stop_words import STOP_WORDS
from spellchecker import SpellChecker
import os


class TransformerInstance:
    # utils to clean the text

    spell = SpellChecker(language="es")
    nlp = spacy.load("es_core_news_sm")
    punctuations = string.punctuation

    # Load the trained model
    model_path = os.path.join(
        settings.BASE_DIR, 'classification', 'models', 'logistic_regression_model.joblib')

    def remove_urls(self, word):
        parsed_text = re.sub(r"\S*https?:\S*", "", word, flags=re.MULTILINE)
        return parsed_text

    def spellcheck_correct(self, text):
        corrected_text = self.spell.correction(text)
        if corrected_text is None:
            corrected_text = text
        return corrected_text

    def spacy_tokenizer(self, text):
        # Create tokens list and correct the spelling of the text
        corrected_text = self.spellcheck_correct(text)
        tokens = self.nlp(corrected_text)

        # Lemmatize and lowercase each token
        tokens = [word.lemma_.lower().strip() if word.lemma_ !=
                  "PROPN" else word.lower_ for word in tokens]

        # Remove stopwords from token list
        tokens = [
            word for word in tokens if word not in STOP_WORDS and word not in self.punctuations]

        # Remove links from token list
        tokens = [self.remove_urls(word) for word in tokens]

        # return the tokens list preprocessed
        return tokens

    class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            return [self.clean_text(text) for text in X]

        def fit(self, X, y=None, **fit_params):
            return self

        def get_params(self, deep=True):
            return {}

    # Basic function to clean the text

    def clean_text(self, text):
        return text.strip().lower()

    def get_model(self):
        logistic_regression_model = load(self.model_path)
        return logistic_regression_model
