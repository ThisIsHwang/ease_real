from app import app
import sys
from konlpy.tag import Okt
import pandas as pd
def getVerbAndNoun(text):
    okt = Okt()
    df = pd.DataFrame(okt.pos(text), columns=['morph', 'tag'])
    df.set_index('tag', inplace=True)
    return df['morph'].values

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, use_reloader=False)