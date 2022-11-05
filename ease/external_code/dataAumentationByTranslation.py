import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/jaecheol/opt/anaconda3/envs/nonmentor/key.json"
import pandas as pd
import random



def list_languages():
    """Lists all available languages."""
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    results = translate_client.get_languages()
    return results

def translate_text(source, target, text, google=None):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate


    translate_client = translate.Client()
    # if isinstance(text, six.binary_type):
    #     text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, source_language=source, target_language=target)
    return result["translatedText"]

def makingAugmentationbackTranslation(text, num):
    languageList = list_languages()
    languages = [lang["language"] for lang in languageList]
    random.shuffle(languages)
    translatedList = []
    translatedResultList = []
    korToJap = translate_text(source="ko", target="ja", text=text)
    japToKor = translate_text(source="ja", target="ko", text=korToJap)

    translatedResultList.append(japToKor)
    index = 0
    while index < len(languages):
        try:
            if num == index:
                break
            l = languages[index]
            if l == "ja":
                index += 1
                continue
            translatedList.append((l, translate_text(source="ko", target=l, text=japToKor)))
            index += 1
        except Exception as e:
            print(e)
            index += 1
            continue

    index = 0
    while index < len(translatedList):
        try:
            l, t = translatedList[index]
            translatedResultList.append((translate_text(source=l, target="ko", text=t)))
            index += 1
        except Exception as e:
            print(e)
            index += 1
            continue
    return translatedResultList
