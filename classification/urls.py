from django.urls import path

from . import views


app_name = 'classification'

urlpatterns = [
    path('', views.index, name='index'),
    path('classify/', views.classify, name='classify'),
    path('classify-file/', views.classify_file, name='classify-file'),
]
