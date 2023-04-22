import tweepy as tw
import pandas as pd
import glob
from classification.connection import TwitterApi
from classification.tranformers import TransformerInstance


def classify_csv_data():
    # Perform the prediction
    files_to_classify = './testing'
    files_path_list = glob.glob(f'{files_to_classify}/*.csv')

    multiple_df = [pd.read_csv(file_path, encoding='utf-8-sig')
                   for file_path in files_path_list]
    dataframe_to_classify = pd.concat(multiple_df)

    # Clean the dataset to delete None values, urls, etc.
    clean_df_to_classify = [" ".join(TransformerInstance.spacy_tokenizer(text))
                            for text in dataframe_to_classify['text']]

    logistic_regression_model = TransformerInstance.get_model()
    predicted_labels = logistic_regression_model.predict(clean_df_to_classify)
    classified_df = pd.DataFrame(
        {'text': dataframe_to_classify['text'], 'label': predicted_labels}
    )
    classified_df.to_csv(
        './classified_dataset/classified_data.csv', index=False)


def classify_tweets_with_twitter_api(api_key, api_secret):
    api = TwitterApi().twitter_api_connection(api_key, api_secret)
    # Search query parameters
    search_query = 'covid -filter:retweets'

    # Get tweets from the API
    tweets = tw.Cursor(api.search, q=search_query, lang="es",
                       since="2020-03-21", until="2020-03-26").items(6000)

    # Store the responses
    tweets_list = []
    for tweet in tweets:
        tweets_list.append(tweet)

    # Verify the length of the list
    print("Total tweets fetched: ", len(tweets_list))

    # Initializing the dataframe
    tweets_df = pd.DataFrame()

    # Populate the dataframe
    for tweet in tweets_list:
        hashtags = []
        try:
            for hashtag in tweet.entities["hashtags"]:
                hashtags.append(hashtag["text"])
            text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
        except:
            continue
        tweets_df = tweets_df.append(pd.DataFrame({
            'user_name': tweet.user.name,
            'user_location': tweet.user.location,
            'user_description': tweet.user.description,
            'user_verified': tweet.user.verified,
            'date': tweet.created_at,
            'text': text,
            'hashtags': [hashtags if hashtags else None],
            'source': tweet.source
        }))
        tweets_df = tweets_df.reset_index(drop=True)

    # Show the dataframe
    tweets_df.head()

    # Clean the dataframe to perform classification
    clean_tweets_df = [" ".join(TransformerInstance.spacy_tokenizer(text))
                       for text in tweets_df['text']]

    # Predict using the model and the clean dataframe
    logistic_regression_model = TransformerInstance.get_model()
    predicted_labels = logistic_regression_model.predict(clean_tweets_df)

    # Save the classified dataset in a new file
    classified_df = pd.DataFrame(
        {'text': tweets_df['text'], 'label': predicted_labels}
    )
    classified_df.to_csv(
        "./classified_dataset/classified_data.csv", index=False)
