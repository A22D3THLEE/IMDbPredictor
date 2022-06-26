from .import views
from django.urls import path


app_name = 'predict'


urlpatterns = [
    path('', views.predict, name='predict'),
    path('predict/', views.predict_chances, name='submit_prediction'),
    path('results/', views.view_result, name='results'),
    path('about/', views.view_info, name='about')
]