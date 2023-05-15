from django.shortcuts import render
import os
from classification.utils import classify_csv_data, classify_tweets_with_twitter_api
from django.http import FileResponse


def index(request):
    return render(request, 'index.html')


def classify(request):
    api_key = request.POST.get('api_key', None)
    api_secret = request.POST.get('api_secret', None)
    search_query = request.POST.get('search_query', None)
    end_date = request.POST.get('end_date', None)

    if not api_key or not api_secret:
        return render(request, 'index.html', {'error': 'Please provide both API key and secret.'})

    if not search_query:
        return render(request, 'index.html', {'error': 'Please provide a search query.'})
    if not end_date:
        return render(request, 'index.html', {'error': 'Please provide end date.'})

    try:
        data = {
            'api_key': api_key,
            'api_secret': api_secret,
            'search_query': search_query,
            'end_date': end_date
        }
        csv_file_path = classify_tweets_with_twitter_api(data)

        response = FileResponse(
            open(csv_file_path, 'rb'), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_file_path)}"'
        return response
    except Exception as e:
        print(e)
        return render(request, 'index.html', {'error': 'Something went wrong. Please try again.'})


def classify_file(request):
    if 'csv_file' not in request.FILES:
        return render(request, 'index.html', {'error': 'Please provide a CSV file.'})

    csv_file = request.FILES['csv_file']

    try:
        csv_file_path = classify_csv_data(csv_file)

        response = FileResponse(
            open(csv_file_path, 'rb'), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_file_path)}"'
        return response
    except Exception as e:
        print(e)
        return render(request, 'index.html', {'error': 'Something went wrong. Please try again.'})
