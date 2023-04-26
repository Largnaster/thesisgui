import tweepy as tw
import pandas as pd
import glob
import os
from django.conf import settings
from django.utils import timezone
from classification.connection import TwitterApi
from manage import spacy_tokenizer, get_model


def classify_csv_data():
    # Perform the prediction
    files_to_classify = './testing'
    files_path_list = glob.glob(f'{files_to_classify}/*.csv')

    multiple_df = [pd.read_csv(file_path, encoding='utf-8-sig')
                   for file_path in files_path_list]
    dataframe_to_classify = pd.concat(multiple_df)

    # Clean the dataset to delete None values, urls, etc.
    clean_df_to_classify = [" ".join(spacy_tokenizer(text))
                            for text in dataframe_to_classify['text']]

    logistic_regression_model = get_model()
    predicted_labels = logistic_regression_model.predict(clean_df_to_classify)
    classified_df = pd.DataFrame(
        {'text': dataframe_to_classify['text'], 'label': predicted_labels}
    )
    classified_df.to_csv(
        './classified_dataset/classified_data.csv', index=False)


def classify_tweets_with_twitter_api(data):
    api = TwitterApi().twitter_api_connection(
        data["api_key"], data["api_secret"])
    # Search query parameters
    search_query = '{search} -filter:retweets'.format(
        search=data["search_query"])

    # Get tweets from the API
    tweets = tw.Cursor(api.search_tweets, q=search_query, lang="es",
                       until=data["end_date"]).items(15)

    # Store the responses
    tweets_list = []
    for tweet in tweets:
        tweets_list.append(tweet)

    # Verify the length of the list
    print("Total tweets fetched: ", len(tweets_list))

    # Populate the dataframe
    tweet_data = []
    for tweet in tweets_list:
        try:
            hashtags = [hashtag["text"]
                        for hashtag in tweet.entities["hashtags"]]
            text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
            tweet_data.append(
                {'user_name': tweet.user.name,
                 'user_location': tweet.user.location,
                 'user_description': tweet.user.description,
                 'user_verified': tweet.user.verified,
                 'date': tweet.created_at,
                 'text': text,
                 'hashtags': hashtags if hashtags else None,
                 'source': tweet.source
                 })
        except:
            continue

    # tweets_df = tweets_df.reset_index(drop=True)
    # Initializing the dataframe
    tweets_df = pd.DataFrame(tweet_data)

    # Show the dataframe
    tweets_df.head()

    # Clean the dataframe to perform classification
    logistic_regression_model = get_model()
    clean_tweets_df = [" ".join(spacy_tokenizer(text))
                       for text in tweets_df['text']]

    # Predict using the model and the clean dataframe
    predicted_labels = logistic_regression_model.predict(clean_tweets_df)

    # Save the classified dataset in a new file
    classified_df = pd.DataFrame(
        {'text': tweets_df['text'], 'label': predicted_labels}
    )
    file_name = 'classified_data {}.csv'.format(
        timezone.now().strftime("%Y-%m-%d %H-%M-%S"))
    output_path = os.path.join(
        settings.BASE_DIR, 'classification', 'classified_dataset', file_name)
    classified_df.to_csv(output_path, index=False)

    return output_path
