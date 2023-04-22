from django.shortcuts import render

from classification.utils import classify_tweets_with_twitter_api


def index(request):
    return render(request, 'index.html')


def classify(request):
    api_key = request.POST.get('api_key', None)
    api_secret = request.POST.get('api_secret', None)
    if not api_key or not api_secret:
        return render(request, 'index.html', {'error': 'Please provide both API key and secret.'})
    print(api_key, api_secret)
    try:
        classify_tweets_with_twitter_api(api_key, api_secret)
    except Exception as e:
        print(e)
        return render(request, 'index.html', {'error': 'Something went wrong. Please try again.'})
    return render(request, 'index.html', {'api_key': api_key, 'api_secret': api_secret})
