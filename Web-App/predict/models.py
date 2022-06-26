# Create your models here.
from django.db import models


class PredResults(models.Model):
    text = models.CharField(max_length=200)
    classification = models.CharField(max_length=30)
    rating = models.IntegerField()


    def __str__(self):
        return self.classification