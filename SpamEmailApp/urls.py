from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),	      
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
	       path("UploadDataset.html", views.UploadDataset, name="UploadDataset"),	      
               path("UploadDatasetAction", views.UploadDatasetAction, name="UploadDatasetAction"),
               path("TrainGA", views.TrainGA, name="TrainGA"),
               path("TrainPSO", views.TrainPSO, name="TrainPSO"),
               path("Graph", views.Graph, name="Graph"),
	       path("SpamDetectionAction", views.SpamDetectionAction, name="SpamDetectionAction"),
               path("SpamDetection.html", views.SpamDetection, name="SpamDetection"),
	       
]
