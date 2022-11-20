# -*- coding: utf-8 -*-
from time import time
start = time()

import joblib
from konlpy.tag import Okt
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

def getVerbAndNoun(text):
    okt = Okt()
    df = pd.DataFrame(okt.pos(text), columns=['morph', 'tag'])
    df.set_index('tag', inplace=True)
    return df['morph'].values

PK_NUM = 69

df = pd.read_csv('problem_2.csv')
essays = df['essay']

model_name = f'/home/ubuntu/anaconda3/envs/nonmentor/ease_real/models/Problem_{PK_NUM}/model.pkl'
model = joblib.load(model_name)

sbert = SentenceTransformer('/home/ubuntu/anaconda3/envs/nonmentor/ease_real/models/SBERT')

from ease.grade import grade

essay1 = '''제시문 (가)는 인물 유형을 신중하게 접근하는 햄릿형과 과감하게 도전하는 돈키호테형으로 분류하고 있다. 이에 따르면 (다)와 (마)의 주체들은 햄릿형, (나), (라)의 주체들은 돈키호테형으로 분류된다. (다)는 장수가 공격과 수비를 신중하게 결정하고, 기미에 맞게 신중하게 대처해야 함을 강조한다. (마)의 리더들은 급변하는 상황에서도 실패에 대한 두려움 때문에 의사결정을 지나치게 신중하게 하고 신속한 집행을 회피한다. 반면 (나)의 피카소는 유명한 화가였음에도 불구하고 안주하지 않고 새로운 영역에 도전하는 모습을 보여준다. (라)의 ‘나’는 탈출을 위해 목숨을 걸고 바다에 몸을 던지는 과감한 도전을 하고 있다.
'''

min_sim = 1e9
for i, essay in enumerate(essays):
    score = grade(model, essay1, PK_NUM, sbert=sbert)
    # print(score['d_score'][0])
    # print(score['n_score'][0])
    # print(score['p_score'][0])
    print(f'{i}: {score["similarity"]}')
    if min_sim > score["similarity"]:
        min_sim = score["similarity"]

end = time()
print(f'경과시간: {end-start}')
