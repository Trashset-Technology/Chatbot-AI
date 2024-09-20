from django.urls import path
from .views import train_chatbot, chat  # Ensure you import the new view

urlpatterns = [
    path("train/", train_chatbot, name="train_chatbot"),
    path("chat/", chat, name="chat"),  
]
