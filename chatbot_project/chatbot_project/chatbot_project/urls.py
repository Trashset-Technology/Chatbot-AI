from django.contrib import admin
from django.urls import path, include
from chatbot.views import *

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("chatbot.urls")),
    path('',home)
]
