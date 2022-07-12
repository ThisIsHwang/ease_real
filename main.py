import os
import sys
import inspect
import pandas as pd
import nltk
sys.path.append("/usr/bin/aspell")
import pickle
import joblib
joblib

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# training_set_rel3.xlsx 를 현재 디렉토리로 옮겨야됨
df = pd.read_csv("training_set_rel3_ko.tsv")

df = df[df['essay_set']==5]
df['score'] = df['domain1_score']
#"짝수" if num % 2 == 0 else "홀수"
df['score']
df
#df = df[['essay_ko', 'score']]

# df["domain1_score"] = df["domain1_score"].astype(int)

prompt = "회고록에서 저자가 만들어낸 분위기를 묘사하라. 회고록의 관련적이고 구체적인 정보로 답변을 뒷받침하세요."

from ease.create import create

# try:
#     model = joblib.load('model.pkl')
# except Exception as e:
#     model = create(df["essay"].tolist(), df["score"].tolist(), prompt)
#

model = create(df["essay_ko"].tolist(), df["score"].tolist(), prompt)
joblib.dump(model, 'model.pkl')
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

print(grade(model, df["essay"].iloc[0]))