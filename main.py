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
# training_set_rel3.xlsx 를 현재 디렉토리로 옮겨야됨
#논술 한글
testSet = pd.read_csv("ebsi_0701_testset.csv")
trainSet = pd.read_csv("ebsi_0701_trainset.csv")
df = trainSet["score"].value_counts()

#한글 ASAP
#df = pd.read_csv("training_set_rel3_ko.tsv")

# df = df[df['essay_set']==1]
# df['score'] = df['domain1_score']
#df = pd.read_csv('/Users/hwangyun/PycharmProjects/ease_real/testfile.csv')

#"짝수" if num % 2 == 0 else "홀수"
#df['score']
#df = df[['essay_ko', "score"]]

# df["domain1_score"] = df["domain1_score"].astype(int)

total_path = '/Users/hwangyun/PycharmProjects/ease_real/lab/total.csv'
prompt_path = '/Users/hwangyun/PycharmProjects/ease_real/lab/prompt.csv'

df = pd.read_csv(total_path)
df = df.dropna()
problem_list = list(pd.unique(df['problem_id']))
prompt_df = pd.read_csv(prompt_path)

import pickle
from ease.create import create
from ease.grade import grade
from ease.util_functions import sub_chars
from sklearn.model_selection import train_test_split
from ease.util_functions import quadratic_weighted_kappa as qwk

qwk_dict = dict()


results = []
for num in problem_list:
  df_focus = df[df['problem_id'] == num]
  p = prompt_df[prompt_df['problem_id']==num]['논제'].tolist()[0]
  p = sub_chars(p)
  train_df, test_df = train_test_split(df_focus, train_size=0.8)
  e = train_df['essay'].tolist()
  s = train_df['총점'].tolist()
  mx = 0
  for j in range(10):
    model = create(e,s,p, lgbm=True)
    e_test = test_df['essay'].tolist()
    a_s = test_df['총점'].tolist()
    p_s = []

    for i in range(len(test_df)):
      result = grade(model, e_test[i])
      p_s.append(result['score'])

    qwk_s = qwk(a_s, p_s)
    results.append(f'Model {num} trial {j}: {qwk(a_s, p_s)}')

    if qwk_s > mx:
      mx = qwk_s
      with open(f'/Users/hwangyun/PycharmProjects/ease_real/lab/m_{num}_v2.pickle', "wb") as f:
        pickle.dump(model, f)

for s in results:
  print(s)
print()




#
# prompt = """(가)와 (나)는 공통적으로 경쟁이 우리 삶에서 떼어낼 수 없는 불가피한 것이라고 인식한다. 그러나 경쟁 발생의 원인과, 경쟁을 수용하는 태도의 측면에서 차이를 갖는다. 원인의 측면에서 (가)는 경쟁심이 인간의 본능이라고 주장하는 반면 (나)는 한정된 자원으로 인해 불가피하게 경쟁이 발생한다고 본다. 경쟁을 수용하는 태도의 측면에서 (가)는 경쟁이 개개인의 의욕과 노력을 더 이끌어내는 긍정적인 상호작용이라고 간주한다. 반면에, (나)는 다양성과 화합이 최대의 가치로 여겨지는 시대가 도래하고 있기 때문에 경쟁보다는 공존을 추구해야 한다고 주장한다."""
#
# essays = trainSet["essay"].tolist()
# scores = trainSet["score"].tolist()
# model = create(essays, scores, prompt, lgbm=True)
# # joblib.dump(model, 'model.pkl')
# # model = joblib.load('model.pkl')
# for key, value in model.items():
#   if key != "text" and key != "score" and key != "prompt":
#     print(key,":",value)
#
# from ease.grade import grade
#
#
#
# predList = []
# for data in testSet["essay"]:
#   data = util_functions.sub_chars(data)
#   score = grade(model, data)
#   predList.append(score["score"])
#   print(score)
#   print("당신의 점수는", score["score"], "점입니다.")
#
# print("kappa_score:", util_functions.quadratic_weighted_kappa(predList, list(testSet["score"])))
# print()
# print(predList)
# print(list(testSet["score"]))
#
# clf = model['classifier']
# f_n = clf.feature_name_
# f_i = clf.feature_importances_
#
# f_dict = dict()
# for n, i in zip(f_n, f_i):
#   f_dict[n] = i
#
# f_list = sorted(f_dict.items(), key=lambda x: x[1], reverse=True)
#
#
#
# from lightgbm import plot_importance
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(model['classifier'], ax=ax)
# plt.show()