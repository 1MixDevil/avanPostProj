from django.urls import path
from .views import index
from .views import index, retrain

urlpatterns = [
    path('', index, name='index'),
    path('test/', index, name='index'),
    path('retrain/', retrain, name='retrain'),
]