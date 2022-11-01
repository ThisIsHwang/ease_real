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

trainSet = pd.read_csv('ebsi_0502_trainset_for_dockhae.csv')
testSet = pd.read_csv('ebsi_0502_testset_for_dockhae.csv')
#"짝수" if num % 2 == 0 else "홀수"
#df['score']
#df = df[['essay_ko', "score"]]

# df["domain1_score"] = df["domain1_score"].astype(int)

# total_path = '/Users/hwangyun/PycharmProjects/ease_real/lab/total.csv'
# prompt_path = '/Users/hwangyun/PycharmProjects/ease_real/lab/prompt.csv'
#
# df = pd.read_csv(total_path)
# df = df.dropna()
# problem_list = list(pd.unique(df['problem_id']))
# prompt_df = pd.read_csv(prompt_path)

import pickle
from ease.create import create
from ease.grade import grade
from ease.util_functions import sub_chars
from sklearn.model_selection import train_test_split
from ease.util_functions import quadratic_weighted_kappa as qwk

qwk_dict = dict()


# results = []
# for num in problem_list:
#   df_focus = df[df['problem_id'] == num]
#   p = prompt_df[prompt_df['problem_id']==num]['논제'].tolist()[0]
#   p = sub_chars(p)
#   train_df, test_df = train_test_split(df_focus, train_size=0.8)
#   e = train_df['essay'].tolist()
#   s = train_df['총점'].tolist()
#   mx = 0
#   for j in range(10):
#     model = create(e,s,p, lgbm=True)
#     e_test = test_df['essay'].tolist()
#     a_s = test_df['총점'].tolist()
#     p_s = []
#
#     for i in range(len(test_df)):
#       result = grade(model, e_test[i])
#       p_s.append(result['score'])
#
#     qwk_s = qwk(a_s, p_s)
#     results.append(f'Model {num} trial {j}: {qwk(a_s, p_s)}')
#
#     if qwk_s > mx:
#       mx = qwk_s
#       with open(f'/Users/hwangyun/PycharmProjects/ease_real/lab/m_{num}_v2.pickle', "wb") as f:
#         pickle.dump(model, f)
#
# for s in results:
#   print(s)
# print()





prompt = """[가]에 제시된 상황을 [나], [다]를 바탕으로 분석하시오. (300자 내외)
[가] 청소년들이 차별과 혐오를 유희처럼 또래문화에서 즐기는 일은 과거에도 있었다. 하지만 스
마트폰의 급격한 보급, 자정과 규제 없는 개인 인터넷 방송의 증가는 우리가 알지 못했던 상자를
열었다. 인터넷 커뮤니티에서 시작된 혐오문화가 10대의 교실을 잠식한 것이다. (…) 모든 아이들
이 혐오를 즐기는 건 아니다. 불편함과 거부감을 호소하는 학생들도 있다. “잘못된 건 다들 알거
든요. 근데 학교는 작은 사회잖아요. 반기를 들면 ‘쟤 이상해’ 이런 취급을 당해요.”, “쿨하고 싶어
서 대응을 잘 못해요. 애들이 친 농담을 웃어넘기고 인정하는 애들이 인기가 많으니까요. 맞장구
치고 같이 키득거리거나 아니면 침묵하거나, 그렇게 되는 거죠.” 혐오 표현이 쿨한 것으로 여겨지
면서, 불편함을 느끼고 상처받는 아이들의 존재는 지워진다.                              
[나] 사회는 구성원이 사회 규범을 지키도록 기대하고 법과 같은 규범을 통해 이를 강제하기도
한다. 그러나 때로 사회 구성원이 이러한 사회적 규범이나 기대에 벗어난 사고와 행동을 하기도
하는데, 이것을 일탈 행동이라고 한다. 일탈 행동은 사회적으로 바람직하지 못한 행동으로 그 사
회의 통합과 존속을 저해하기도 하지만 때로 그 사회의 문제를 표출함으로써 사회 변화의 계기
가 될 수도 있다. (…) 뒤르켐에 따르면 개인은 한 사회의 규범을 행동의 지침으로 여기는데, 사
회 규범이 약화되거나 주도적 규범이 없는 상태가 되면 일탈 행동을 하게 된다. (…) 차별 교제
이론은 일탈 행동을 하는 집단이나 사람들과의 접촉을 통해 일탈 행동이 학습된다는 입장이다.
즉, 일탈 행동을 보이는 사람들과 접촉하는 과정에서 그들과 동화되어 일탈 행동을 하게 된다는
것이다.
                                             
[다] 심리 실험이 진행되었다. 참가자는 한 명을 제외하면 모두 가짜였다. 실험자가 직선 하나가
그려진 카드를 보여준 후, 길이가 다른 직선 세 개가 그려진 다른 카드를 참가자들에게 보여주었
다. 두 번째 카드의 직선 중 하나는 처음에 제시한 카드의 직선과 길이가 같았는데, 그 직선을
고르는 것이 참가자들에게 주어진 과제였다. 참가자는 한 사람씩 큰 소리로 자신의 선택을 말했
다. 진짜 참가자는 끝자리에 앉았기 때문에 앞사람들의 대답을 알 수 있었다. 가짜 참가자들은 일
부러 오답을 선택했다. 당황하는 기색이 역력했던 진짜 참가자의 선택은 두 가지였다. 자신의 선
택을 유지하면서 혼자 다른 대답을 하거나, 다른 참가자들과 같은 대답을 하는 것이었다. (…) 이
런 실험을 수십 번 진행했는데, 다른 이들이 오답을 말할 때 진짜 참가자가 자신의 의견을 관철
하는 경우는 37%였다. 그 외는 매번 다수의 의견을 따랐다.
 실험 종료 후 진짜 참가자에게 실험에 대해 사실대로 설명하자, 안도의 한숨을 내쉬며 이렇게
말했다. “그들이나 저, 둘 중 한쪽은 비정상이었어요. 그들처럼 제 판단력도 형편없는지 궁금했지
만 그들이 맞지 않을까 싶은 생각도 들어서 결정을 못 내렸죠.”, “그들이 맞아서가 아니라 그저
묻어가려고 한 거예요. 반대 의견을 펴려면 엄청난 배짱이 필요한 것 같아요.”"""

essays = trainSet["essay"].tolist()
scores = trainSet["score"].tolist()


model_name = 'model_for_dockhae.pkl'
if os.path.isfile(model_name):
    model = joblib.load(model_name)
else:
    model = create(essays, scores, prompt, lgbm=True, generate_additional=False)
    joblib.dump(model, model_name)


# model = joblib.load('model.pkl')
for key, value in model.items():
  if key != "text" and key != "score" and key != "prompt":
    print(key,":",value)

from ease.grade import grade

# predList = []
# for data in testSet["essay"]:
#   data = util_functions.sub_chars(data)
#   score = grade(model, data)
#   predList.append(score["score"])
#   print(score)
#   print("당신의 점수는", score["score"], "점입니다.")
#
# # data = util_functions.sub_chars(data)
# score = grade(model, "씨발 몰라 개새끼야")
# print(score)
# print("kappa_score:", util_functions.quadratic_weighted_kappa(predList, list(testSet["score"])))
# print()
# print(predList)
# print(list(testSet["score"]))

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

if False:
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
plt.rc('font', family='AppleGothic')
plot_importance(model['classifier'], ax=ax)
plt.show()

print(1)