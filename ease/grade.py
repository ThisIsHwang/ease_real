"""
Functions to score specified data using specified ML models
"""

import sys
import pickle
import os
import numpy
import logging
from konlpy.tag import Okt
okt=Okt()

#Append sys to base path to import the following modules
base_path = os.path.dirname(__file__)
sys.path.append(base_path)

#Depend on base path to be imported
from .essay_set import EssaySet
from . import predictor_extractor
from . import predictor_set
from . import util_functions

#Imports needed to unpickle grader data
from . import feature_extractor
import sklearn.ensemble
import math

log = logging.getLogger(__name__)
fword_path = "/home/ubuntu/anaconda3/envs/nonmentor/ease_real/ease/data/fword_list.txt"
fwordList = []

with open(fword_path, "r") as file:
    fwordList = file.readlines()
    fwordList = list(map(lambda x: x.strip(), fwordList))


def isFword(answer):
    #print(fwordList)
    answer = okt.nouns(answer)
    #
    fwordFlag = False
    for a in answer:
        if a in fwordList:
            fwordFlag = True
            break
    if fwordFlag:
        return True
    else:
        return False



def grade(grader_data, submission, PK_NUM, sbert=None):
    """
    Grades a specified submission using specified models
    grader_data - A dictionary:
    {
        'models' : trained models,
        'extractor' : trained feature extractor,
        'prompt' : prompt for the question,
        'algorithm' : algorithm for the question,
    }
    submission - The student submission (string)
    """
    results = {'errors': [], 'tests': [], 'd_score': [], 'n_score': [], 'p_score': [], 'feedback': "", 'success': False,
               'confidence': 0}
    #print(submission)
    if isFword(submission):
        results["d_score"].append(0)
        results["n_score"].append(0)
        results["p_score"].append(0)
        results["success"] = True
        return results
    #Initialize result dictionary

    has_error=False

    grader_set=EssaySet(essaytype="test")
    feedback = {}

    d_model, n_model, p_model, extractor = get_classifier_and_ext(grader_data)

    #This is to preserve legacy functionality
    if 'algorithm' not in grader_data:
        grader_data['algorithm'] = util_functions.AlgorithmTypes.classification

    try:
        #Try to add essay to essay set object
        if isinstance(submission, str):
            grader_set.add_essay(str(submission), 0, 0, 0)
            grader_set.update_prompt(str(grader_data['prompt']))
            submission = [submission]
        else:
            for essay in submission:
                grader_set.add_essay(str(essay), 0, 0, 0)
            grader_set.update_prompt(str(grader_data['prompt']))
    except Exception:
        error_message = "Essay could not be added to essay set:{0}".format(submission)
        log.exception(error_message)
        results['errors'].append(error_message)
        has_error=True

    #Try to extract features from submission and assign score via the models
    try:
        grader_feats, sim_matrix = extractor.gen_feats(grader_set, PK_NUM, sbert=sbert, grade=True)
        # feedback=extractor.gen_feedback(grader_set,grader_feats)[0]
        for i in range(len(submission)):
            results['d_score'].append(int(d_model.predict(grader_feats)[i]))
            results['n_score'].append(int(n_model.predict(grader_feats)[i]))
            results['p_score'].append(int(p_model.predict(grader_feats)[i]))
    except Exception:
        error_message = "Could not extract features and score essay."
        log.exception(error_message)
        results['errors'].append(error_message)

    total_similarity = int(sim_matrix.max(axis=1).mean(axis=0) * 100)
    if total_similarity <= 0:
        total_similarity = 0

    if total_similarity <= 45:
        results['d_score'] = [0]
        results['n_score'] = [0]
        results['p_score'] = [0]
    results['similarity'] = total_similarity

    return results

def grade_generic(grader_data, numeric_features, textual_features):
    """
    Grades a set of numeric and textual features using a generic models
    grader_data -- dictionary containing:
    {
        'algorithm' - Type of algorithm to use to score
    }
    numeric_features - list of numeric features to predict on
    textual_features - list of textual feature to predict on

    """
    results = {'errors': [],'tests': [],'score': 0, 'success' : False, 'confidence' : 0}

    has_error=False

    #Try to find and load the models file

    grader_set=predictor_set.PredictorSet(essaytype="test")

    model, extractor = get_classifier_and_ext(grader_data)

    #Try to add essays to essay set object
    try:
        grader_set.add_row(numeric_features, textual_features,0)
    except Exception:
        error_msg = "Row could not be added to predictor set:{0} {1}".format(numeric_features, textual_features)
        log.exception(error_msg)
        results['errors'].append(error_msg)
        has_error=True

    #Try to extract features from submission and assign score via the models
    try:
        grader_feats=extractor.gen_feats(grader_set)
        results['score']=model.predict(grader_feats)[0]
    except Exception:
        error_msg = "Could not extract features and score essay."
        log.exception(error_msg)
        results['errors'].append(error_msg)
        has_error=True

    #Try to determine confidence level
    try:
        results['confidence'] = get_confidence_value(grader_data['algorithm'],model, grader_feats, results['score'])
    except Exception:
        #If there is an error getting confidence, it is not a show-stopper, so just log
        log.exception("Problem generating confidence value")

    if not has_error:
        results['success'] = True

    return results

def get_confidence_value(algorithm,model,grader_feats,score, scores):
    """
    Determines a confidence in a certain score, given proper input parameters
    algorithm- from util_functions.AlgorithmTypes
    models - a trained models
    grader_feats - a row of features used by the models for classification/regression
    score - The score assigned to the submission by a prior models
    """
    min_score=min(numpy.asarray(scores))
    max_score=max(numpy.asarray(scores))
    if algorithm == util_functions.AlgorithmTypes.classification and hasattr(model, "predict_proba"):
        #If classification, predict with probability, which gives you a matrix of confidences per score point
        raw_confidence=model.predict_proba(grader_feats)[0, int(float(score)-float(min_score))]
        #TODO: Normalize confidence somehow here
        confidence=raw_confidence
    elif hasattr(model, "predict"):
        raw_confidence = model.predict(grader_feats)[0]
        confidence = max(float(raw_confidence) - math.floor(float(raw_confidence)), math.ceil(float(raw_confidence)) - float(raw_confidence))
    else:
        confidence = 0

    return confidence

def get_classifier_and_ext(grader_data):
    if 'classifier' in grader_data:
        d_model = grader_data['d_classifier']
        n_model = grader_data['n_classifier']
        p_model = grader_data['p_classifier']
    elif 'models' in grader_data:
        d_model = grader_data['d_models']
        n_model = grader_data['n_models']
        p_model = grader_data['p_models']
    else:
        raise Exception("Cannot find a valid models.")

    if 'feature_ext' in grader_data:
        extractor = grader_data['feature_ext']
    elif 'extractor' in grader_data:
        extractor = grader_data['extractor']
    else:
        raise Exception("Cannot find the extractor")

    return d_model, n_model, p_model, extractor


