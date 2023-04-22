import tweepy as tw


class TwitterApi:
    instance = None

    def __init__(self) -> None:
        if not TwitterApi.instance:
            TwitterApi.instance = self

    def twitter_api_connection(self, api_key, api_secret):
        # Authentication
        auth = tw.OAuthHandler(api_key, api_secret)
        api = tw.API(auth, wait_on_rate_limit=True)
        return api
