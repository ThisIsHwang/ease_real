import os
import sys
import inspect
import pandas as pd
import nltk
sys.path.append("/usr/bin/aspell")
import pickle
import joblib
joblib

# training_set_rel3.xlsx 를 현재 디렉토리로 옮겨야됨
df = pd.read_csv("training_set_rel3_ko.tsv")

df = df[df['essay_set']==1]
df['score'] = df['domain1_score']
#"짝수" if num % 2 == 0 else "홀수"
df['score']
df
#df = df[['essay_ko', 'score']]

# df["domain1_score"] = df["domain1_score"].astype(int)

prompt = "점점 더 많은 사람들이 컴퓨터를 사용하지만 모든 사람이 이것이 사회에 도움이 된다는 데 동의하는 것은 아닙니다. 기술 발전을 지지하는 사람들은 컴퓨터가 사람들에게 긍정적인 영향을 미친다고 믿습니다. 그들은 손과 눈의 협응을 가르치고, 사람들에게 먼 곳과 사람에 대해 배울 수 있는 능력을 부여하고, 사람들이 다른 사람들과 온라인으로 대화할 수 있도록 합니다. 다른 사람들은 다른 생각을 가지고 있습니다. 일부 전문가들은 사람들이 컴퓨터에 너무 많은 시간을 할애하고 운동하고 자연을 즐기고 가족 및 친구들과 교류하는 시간이 적다고 우려합니다.\
컴퓨터가 사람들에게 미치는 영향에 대한 당신의 의견을 담은 편지를 지역 신문에 쓰십시오. 독자들에게 당신의 의견에 동의하도록 설득하십시오."

from ease.create import create

essays = df["essay_ko"].tolist()
scores = df["score"].tolist()
model = create(essays, scores, prompt)
joblib.dump(model, 'model.pkl')
#model = joblib.load('model.pkl')
for key, value in model.items():
  if key != "text" and key != "score" and key != "prompt":
    print(key,":",value)


from ease.grade import grade

# grader_data = \
#     {
#         'model' : model['classifier'],
#         'extractor' : model['feature_ext'],
#         'prompt' : prompt,
#         'algorithm' : model['algorithm'],
#     }

# for i in range(100):
#   tempDf = df["essay_ko"].iloc[i]
#   print(grade(model, tempDf))
#   scoreDf = df["score"].iloc[i]
#   print(grade(model, tempDf)["score"], scoreDf)

a = "컴퓨터는 병신이여서 아무 것도 못한다. 컴퓨터는 무조건 없어져야 한다. 나는 컴퓨터가 매우 싫다."
print("제시문:", prompt)
print("답안:", a)
print("당신의 점수는", grade(model, a)["score"], "점입니다.")
print()