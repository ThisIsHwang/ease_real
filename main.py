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
testSet = pd.read_csv("ebsi_0701_testset_300.csv")
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

prompt = """(가)와 (나)는 공통적으로 경쟁이 우리 삶에서 떼어낼 수 없는 불가피한 것이라고 인식한다. 그러나 경쟁 발생의 원인과, 경쟁을 수용하는 태도의 측면에서 차이를 갖는다. 원인의 측면에서 (가)는 경쟁심이 인간의 본능이라고 주장하는 반면 (나)는 한정된 자원으로 인해 불가피하게 경쟁이 발생한다고 본다. 경쟁을 수용하는 태도의 측면에서 (가)는 경쟁이 개개인의 의욕과 노력을 더 이끌어내는 긍정적인 상호작용이라고 간주한다. 반면에, (나)는 다양성과 화합이 최대의 가치로 여겨지는 시대가 도래하고 있기 때문에 경쟁보다는 공존을 추구해야 한다고 주장한다."""



essays = trainSet["essay"].tolist()
scores = trainSet["score"].tolist()
model = create(essays, scores, prompt)
# joblib.dump(model, 'model.pkl')
# model = joblib.load('model.pkl')
for key, value in model.items():
  if key != "text" and key != "score" and key != "prompt":
    print(key,":",value)


from ease.grade import grade



predList = []
for data in testSet["essay"]:
  data = util_functions.sub_chars(data)
  score = grade(model, data)
  predList.append(score["score"])
  print(score)
  print("당신의 점수는", score["score"], "점입니다.")

print("kappa_score:", util_functions.quadratic_weighted_kappa(predList, list(testSet["score"])))
print()
print(predList)
print(list(testSet["score"]))