from flask import Flask
from flask import jsonify, request
import os

app = Flask(__name__)

import joblib
import sys
from konlpy.tag import Okt
import pandas as pd

def getVerbAndNoun(text):
    okt = Okt()
    df = pd.DataFrame(okt.pos(text), columns=['morph', 'tag'])
    df.set_index('tag', inplace=True)
    return df['morph'].values

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/ubuntu/anaconda3/envs/nonmentor/ease_real")


from ease.grade import grade
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('/home/ubuntu/anaconda3/envs/nonmentor/ease_real/models/SBERT')

@app.route('/getScore', methods=['POST'])
def index():
    answer = request.form.get('answer')
    pk = str(request.form.get('pk'))
    path_dir = f'/home/ubuntu/anaconda3/envs/nonmentor/ease_real/models/Problem_{pk}/model.pkl'
    model = joblib.load(path_dir)
    #print(model)
    scores = grade(model, answer, pk, sbert=sbert)
    print(scores)

    dockhaeScore = scores["d_score"][0]
    nonliScore = scores["n_score"][0]
    pyohyunScore = scores["p_score"][0]
    similarity = scores['similarity']
    print(similarity)

    data = {
        "chongjumScore" : dockhaeScore+ nonliScore + pyohyunScore, #dockhaeScore["score"] + nonliScore["score"] + pyohyunScore["score"],
        "dockhaeScore" : dockhaeScore,#dockhaeScore["score"],
        "nonliScore" : nonliScore,#["score"],
        "pyohyunScore": pyohyunScore,#["score"],
        "similarity": similarity,
        }
    return jsonify(data)

@app.route('/data')
def data():
   data = {"names": ["John", "Jacob", "Julie", "Jennifer"]}
   return jsonify(data)

