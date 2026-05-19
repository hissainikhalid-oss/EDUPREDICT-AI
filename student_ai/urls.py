from django.contrib import admin
from django.urls import path
from django.http import HttpResponse

def home(request):
    return HttpResponse("EDUPREDICT-AI is Live on Render 🚀")

urlpatterns = [
    path('', home),
    path('admin/', admin.site.urls),
]