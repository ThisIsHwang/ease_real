# -*- coding: utf-8 -*-

import os
import inspect
import pandas as pd
import nltk
import pickle
import joblib
import ease.external_code.dataAumentationByTranslation
#from ease import util_functions
from ease.create import create
from ease import util_functions
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt

from ease.external_code.summary_compare import Comparing

def getVerbAndNoun(text):
    okt = Okt()
    df = pd.DataFrame(okt.pos(text), columns=['morph', 'tag'])
    df.set_index('tag', inplace=True)
    return df['morph'].values

# a = Comparing()
# b = a.predict('안녕하시요')
#
# print(1)


#논술 한글
# testSet = pd.read_csv("ebsi_0701_testset.csv")
# trainSet = pd.read_csv("ebsi_0701_trainset.csv")
# df = trainSet["score"].value_counts()

#한글 ASAP
#df = pd.read_csv("training_set_rel3_ko.tsv")

# df = df[df['essay_set']==1]
# df['score'] = df['domain1_score']
# df = pd.read_csv('ebsi_0502_trainset.csv')
# test = pd.read_csv('ebsi_0502_testset.csv')
#df["총점"] = df["총점"].fillna(2)
#X_train, X_test, y_train, y_test = train_test_split(df["essay_cleaned"].tolist(), df["총점"].tolist(), test_size=0.2, random_state=42)

# trainSet = pd.read_csv('ebsi_0502_trainset_for_dockhae.csv')
# testSet = pd.read_csv('ebsi_0502_testset_for_dockhae.csv')
#"짝수" if num % 2 == 0 else "홀수"
#df['score']
#df = df[['essay_ko', "score"]]

# df["domain1_score"] = df["domain1_score"].astype(int)

df = pd.read_csv('problem_2.csv')

trainSet, testSet = train_test_split(df, test_size=0.2, random_state=42)
trainSet = df
# X_train, X_test, Y_train, Y_test = train_test_split(df['essay'].tolist(), df['D'].tolist(), test_size=0.2, random_state=42)
# trainSet = pd.DataFrame({'essay': X_train, 'score': Y_train})
# testSet = pd.DataFrame({'essay': X_test, 'score': Y_test})


import pickle
from ease.create import create
from ease.grade import grade
from ease.util_functions import sub_chars
from sklearn.model_selection import train_test_split
from ease.util_functions import quadratic_weighted_kappa as qwk

qwk_dict = dict()

root_path = '/home/ubuntu/anaconda3/envs/nonmentor/ease_real/models/'
problem_info = os.path.join(root_path, 'Problem_2')

with open(os.path.join(problem_info, 'prompt.txt'), 'r') as f:
    prompt = f.readline()

essays = trainSet["essay"].tolist()
d_score = trainSet["D"].tolist()
n_score = trainSet["N"].tolist()
p_score = trainSet["P"].tolist()

save_path = '/home/ubuntu/anaconda3/envs/nonmentor/ease_real/models/Problem_2'
model_name = os.path.join(save_path, 'model.pkl')
if os.path.isfile(model_name):
    model = joblib.load(model_name)
else:
    model = create(essays, d_score, n_score, p_score, prompt, model_type='xgboost', generate_additional=True)
    joblib.dump(model, model_name)



# models = joblib.load('models.pkl')
for key, value in model.items():
  if key != "text" and key != "score" and key != "prompt":
    print(key,":",value)

from ease.grade import grade

score = grade(model, testSet['essay'])
d_predList = score["d_score"]
n_predList = score["n_score"]
p_predList = score["p_score"]


# for data in testSet["essay"]:
#   data = util_functions.sub_chars(data)
#   score = grade(model, data)
#   d_predList = score["d_score"]
#   n_predList = score["n_score"]
#   p_predList = score["p_score"]
#   print(score)
#   print("당신의 점수는", score["score"], "점입니다.")

import time

# data = util_functions.sub_chars(data)
start = time.time()
score = grade(model, "씨발 몰라 개새끼야")
print(score)
end = time.time()
print(end-start)
print()


print("D kappa_score:", util_functions.quadratic_weighted_kappa(d_predList, list(testSet["D"])))
print(d_predList)
print(list(testSet["D"]))
print()

print("N kappa_score:", util_functions.quadratic_weighted_kappa(n_predList, list(testSet["N"])))
print(n_predList)
print(list(testSet["N"]))
print()

print("P kappa_score:", util_functions.quadratic_weighted_kappa(p_predList, list(testSet["P"])))
print(p_predList)
print(list(testSet["P"]))
print()
clf = model['classifier']
f_n = clf.feature_name_
f_i = clf.feature_importances_

f_dict = dict()
for n, i in zip(f_n, f_i):
  f_dict[n] = i

f_list = sorted(f_dict.items(), key=lambda x: x[1], reverse=True)

from konlpy.tag import Okt
from sklearn.model_selection import train_test_split

# file_path = ''
# df = pd.read_csv(file_path)
# train_df, test_df = train_test_split(df, train_size=0.8)

aa = 0
if aa:
    e_train = trainSet["essay"].tolist()
    e_test = testSet["essay"].tolist()
    set_train = set()
    okt = Okt()
    for e in e_train:
      temp_set = set(okt.morphs(e))
      set_train = set_train | temp_set
    set_test = set()
    okt = Okt()
    for e in e_test:
      temp_set = set(okt.morphs(e))
      set_test = set_test | temp_set
    intersec = set_train & set_test
    set_train_stem = set()
    okt = Okt()
    for e in e_train:
      temp_set = set(okt.morphs(e, stem=True))
      set_train_stem = set_train_stem | temp_set
    set_test_stem = set()
    okt = Okt()
    for e in e_test:
      temp_set = set(okt.morphs(e, stem=True))
      set_test_stem = set_test_stem | temp_set
    intersec_stem = set_train_stem & set_test_stem
    print(f'Normal vocab:\n train: {len(set_train)} test: {len(set_test)} intersection: {len(intersec)} OOV percent: {(len(set_test)-len(intersec))/len(set_test)*100}')
    print(f'Stem vocab:\n train: {len(set_train_stem)} test: {len(set_test_stem)} intersection: {len(intersec_stem)} OOV percent: {(len(set_test_stem)-len(intersec_stem))/len(set_test_stem)*100}')

from lightgbm import plot_importance
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10, 12))
# df_feature_importance = (
#     pd.DataFrame({
#         'feature': f_n,
#         'importance': f_i,
#     })
#     .sort_values('importance', ascending=False)
# )
# print(df_feature_importance)
plt.rc('font', family=['NanumGothic', ])
plot_importance(model['classifier'], ax=ax)
plt.show()

print(1)