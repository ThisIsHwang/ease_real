# -*- coding: utf-8 -*-
import joblib

model_name = 'model_jae.pkl'
model = joblib.load(model_name)

from ease.grade import grade

score = grade(model, "씨발 몰라 개새끼야")
print(score['score'])
