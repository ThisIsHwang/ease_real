"""
Functions that create a machine learning models from training data
"""

import os
import sys
import logging
import numpy
import joblib

#Define base path and add to sys path
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//'))
sys.path.append(one_up_path)

#Import modules that are dependent on the base path
from . import model_creator
from . import util_functions
from . import predictor_set
from . import predictor_extractor
from datetime import datetime
import json


#Make a log
log = logging.getLogger(__name__)

def dump_input_data(text, score):
    try:
        file_path = base_path + "/tests/data/json_data/"
        time_suffix = datetime.now().strftime("%H%M%S%d%m%Y")
        prefix = "test-case-"
        filename = prefix + time_suffix + ".json"
        json_data = []
        for i in range(0, len(text)):
            json_data.append({'text' : text[i], 'score' : score[i]})
        with open(file_path + filename, 'w+') as outfile:
            json.dump(json_data, outfile)
    except:
        error = "Could not dump data to file."
        log.exception(error)

class FinalModel(object):
    def __init__(self):
        self.f_ext = None
        self.d_clf = None
        self.n_clf = None
        self.p_clf = None

def create(text, d_score, n_score, p_score, prompt_string, model_type=False, generate_additional=True, num=2):
    """
    Creates a machine learning models from input text, associated scores, a prompt, and a path to the models
    TODO: Remove models path argument, it is needed for now to support legacy code
    text - A list of strings containing the text of the essays
    score - a list of integers containing score values
    prompt_string - the common prompt for the set of essays
    """

    algorithm = select_algorithm(d_score) #알고리즘을 선택함 3개 이하면 분류, 3개 이상이면 회귀 알고리즘을 선택
    #Initialize a results dictionary to return
    results = {'errors': [],'success' : False, 'cv_kappa' : 0, 'cv_mean_absolute_error': 0,
               'feature_ext' : "", 'classifier' : "", 'algorithm' : algorithm,
               'd_score' : d_score, 'text' : text, 'prompt' : prompt_string}

    if len(text)!=len(d_score): #텍스트와 점수 리스트의 길이를 확인하여 무결성 체크
        msg = "Target and text lists must be same length."
        results['errors'].append(msg)
        log.exception(msg)
        return results

    try:
        #Create an essay set object that encapsulates all the essays and alternate representations (tokens, etc)
        # root_path = f'/home/ubuntu/anaconda3/envs/nonmentor/ease_real/problems/Problem_{num}'
        # if os.path.isfile(os.path.join(root_path, 'e_set.pkl')):
        #     e_set = joblib.load(os.path.join(root_path, 'e_set.pkl'))
        # else:
        e_set = model_creator.create_essay_set(text, d_score, n_score, p_score, prompt_string, generate_additional=generate_additional)
            # joblib.dump(e_set, f'/home/ubuntu/anaconda3/envs/nonmentor/ease_real/problems/Problem_{num}/e_set.pkl')
    except Exception as e:
        print(e)
        msg = "essay set creation failed."
        results['errors'].append(msg)
        log.exception(msg)

    try:
        #Gets features from the essay set and computes error
        model = FinalModel()
        model.f_ext, model.d_clf, model.n_clf, model.p_clf, d_cv_error_results, n_cv_error_results, p_cv_error_results = model_creator.extract_features_and_generate_models(e_set, algorithm = algorithm, model_type=model_type)

        # feature_ext, classifier, cv_error_results = model_creator.extract_features_and_generate_model(e_set, algorithm = algorithm, model_type=model_type)
        results['d_cv_kappa']=d_cv_error_results['kappa']
        results['d_cv_mean_absolute_error']=d_cv_error_results['mae']
        results['n_cv_kappa'] = n_cv_error_results['kappa']
        results['n_cv_mean_absolute_error'] = n_cv_error_results['mae']
        results['p_cv_kappa'] = p_cv_error_results['kappa']
        results['p_cv_mean_absolute_error'] = p_cv_error_results['mae']
        results['feature_ext']=model.f_ext
        results['d_classifier']=model.d_clf
        results['n_classifier']=model.n_clf
        results['p_classifier']=model.p_clf
        results['algorithm'] = algorithm
        results['success']=True
    except:
        msg = "feature extraction and models creation failed."
        results['errors'].append(msg)
        log.exception(msg)

    return results


def create_generic(numeric_values, textual_values, target, algorithm = util_functions.AlgorithmTypes.regression):
    """
    Creates a models from a generic list numeric values and text values
    numeric_values - A list of lists that are the predictors
    textual_values - A list of lists that are the predictors
    (each item in textual_values corresponds to the similarly indexed counterpart in numeric_values)
    target - The variable that we are trying to predict.  A list of integers.
    algorithm - the type of algorithm that will be used
    """

    algorithm = select_algorithm(target)
    #Initialize a result dictionary to return.
    results = {'errors': [],'success' : False, 'cv_kappa' : 0, 'cv_mean_absolute_error': 0,
               'feature_ext' : "", 'classifier' : "", 'algorithm' : algorithm}

    if len(numeric_values)!=len(textual_values) or len(numeric_values)!=len(target):
        msg = "Target, numeric features, and text features must all be the same length."
        results['errors'].append(msg)
        log.exception(msg)
        return results

    try:
        #Initialize a predictor set object that encapsulates all of the text and numeric predictors
        pset = predictor_set.PredictorSet(essaytype="train")
        for i in range(0, len(numeric_values)):
            pset.add_row(numeric_values[i], textual_values[i], target[i])
    except:
        msg = "predictor set creation failed."
        results['errors'].append(msg)
        log.exception(msg)

    try:
        #Extract all features and then train a classifier with the features
        feature_ext, classifier, cv_error_results = model_creator.extract_features_and_generate_model_predictors(pset, algorithm)
        results['cv_kappa']=cv_error_results['kappa']
        results['cv_mean_absolute_error']=cv_error_results['mae']
        results['feature_ext']=feature_ext
        results['classifier']=classifier
        results['success']=True
    except:
        msg = "feature extraction and models creation failed."
        results['errors'].append(msg)
        log.exception(msg)

    return results

def select_algorithm(score_list):
    #Decide what algorithm to use (regression or classification)
    try:
        #Count the number of unique score points in the score list
        if len(util_functions.f7(list(score_list)))>5:
            algorithm = util_functions.AlgorithmTypes.regression
        else:
            algorithm = util_functions.AlgorithmTypes.classification
    except:
        algorithm = util_functions.AlgorithmTypes.regression

    return algorithm
