import time

import requests
import re
import json



correction_table_re = re.compile(
    u"""<DIV id='correctionTable' style='display:none; width:300px; height:200px;'>(.+)<DIV id='correctionTable2' style='display:none; width:300px; height:200px;'>(.+)</body>""",
    re.MULTILINE|re.DOTALL
    )
underline_re = re.compile(
    u"""<div id='bufUnderline' style='display:none; width:300px; height:200px;'>(.+)</div>""",
    re.MULTILINE|re.DOTALL
    )

error_word_re = re.compile(
    u"""<font id='ul_(\\d+)' color='([#a-zA-Z0-9]+)' class='ul' onclick="fShowHelp\('\d+'\)" >([^<]+)</font>""",
    re.MULTILINE
    )

replace_re = re.compile(
    u"""<TD id='tdErrorWord_\\d+' class='tdErrWord' style='color:[#a-zA-Z0-9]+;' >(.+?)</TD>.+?<TD id='tdReplaceWord_\\d+' class='tdReplace' >(.+?)</TD>.+?<TD id='tdHelp_\\d+?' class='tdETNor'>(.+?)</TD>""",
    re.MULTILINE|re.DOTALL
    )

splitter_re = re.compile(u"(\s+)", re.MULTILINE)

def unescape(origin):
    return origin.replace("&amplt", "<").replace("&ampgt", ">").replace("&amp", "&")


def speller(origin, encoding="utf-8"):
    """입력된 문자열에서 맞춤법이 잘못된 부분을 찾아서 반환한다.
    :param origin: 맞춤법을 검사할 문자열
    :type origin: :class:`basestring`
    :param encoding: (origin이 string일 경우) origin의 인코딩
    :type encoding: :class:`basestring`
    :returns:
            newString : 새롭게 교정된 문자열
            errorCnt : 틀린 맞춤법의 수
    """

    #splitted_origin = splitter_re.split(origin)

    error_words = []

    start_pos = 0  # 텍스트의 누적 시작위치. LCS등을 사용하지 않아도 되도록 트리키하게 접근.

    # for idx in range(0, len(splitted_origin), 600):
    #     target = "".join(splitted_origin[idx:idx + 600])
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}

    params = {'text1': origin}
    while True:
        try:
            r = requests.post("http://speller.cs.pusan.ac.kr/results", data=params, headers=headers)
            if r.status_code == 200:
                break
            time.sleep(3)
        except Exception as e:
            print(e)
            time.sleep(3)
            continue

    raw_result = r.content.decode('utf-8')
    res = re.findall('\[{"str":[가-힣 a-zA-Z:[{()}\S..",]*', raw_result)
    if res == []:
        return params['text1'], 0
    res = res[0][:-1]

    jsonData = json.loads(res)
    WordCount = 0
    newString = jsonData[0]['str']
    for err in jsonData[0]['errInfo']:
        frontSentence = newString[:err['start'] + WordCount]
        corretedWord = err['candWord']
        corretedWordList = corretedWord.split("|")
        backSentence = newString[err['end'] + WordCount:]
        WordCount += len(corretedWordList[0]) - len(err['orgStr'])
        newString = "".join([frontSentence, corretedWordList[0], backSentence])

    errorCnt = len(jsonData[0]['errInfo'])
    # dict = json.loads(res)
    return newString, errorCnt

