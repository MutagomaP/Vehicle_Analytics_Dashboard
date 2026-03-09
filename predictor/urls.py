from predictor import views
from django.urls import path
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(url="data_exploration/", permanent=False), name="home"),
    path("data_exploration/", views.data_exploration_view, name="data_exploration"),
    path("regression_analysis/", views.regression_analysis, name="regression_analysis"),
    path("classification_analysis/", views.classification_analysis, name="classification_analysis"),
    path("clustering_analysis/", views.clustering_analysis, name="clustering_analysis"),
]