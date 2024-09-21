from django.urls import  path
from core.views import  index,chatbot_response

urlpatterns = [
    path("",index,name='index'),
    path('api/chat/', chatbot_response,name='chatbot_response')
]