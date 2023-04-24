from django.shortcuts import render

from classification.utils import classify_tweets_with_twitter_api


def index(request):
    return render(request, 'index.html')


def classify(request):
    api_key = request.POST.get('api_key', None)
    api_secret = request.POST.get('api_secret', None)
    search_query = request.POST.get('search_query', None)
    start_date = request.POST.get('start_date', None)
    end_date = request.POST.get('end_date', None)

    if not api_key or not api_secret:
        return render(request, 'index.html', {'error': 'Please provide both API key and secret.'})

    if not search_query:
        return render(request, 'index.html', {'error': 'Please provide a search query.'})
    if not start_date or not end_date:
        return render(request, 'index.html', {'error': 'Please provide both start and end dates.'})

    print(api_key, api_secret, search_query, start_date, end_date)

    try:
        data = {
            'api_key': api_key,
            'api_secret': api_secret,
            'search_query': search_query,
            'end_date': end_date
        }
        classify_tweets_with_twitter_api(data)
        return render(request, 'index.html', {'results': 'Classification done successfully.'})
    except Exception as e:
        print(e)
        return render(request, 'index.html', {'error': 'Something went wrong. Please try again.'})
