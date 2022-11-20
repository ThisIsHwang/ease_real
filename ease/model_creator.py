#Provides interface functions to create and save models
import lightgbm
import numpy
import re
import nltk
import sys

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import sklearn.ensemble
from itertools import chain

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

from .essay_set import EssaySet
from . import util_functions
from . import feature_extractor
import logging
from . import predictor_extractor
from xgboost import XGBClassifier, XGBRegressor

log=logging.getLogger()

def read_in_test_data(filename):
    """
    Reads in test data file found at filename.
    filename must be a tab delimited file with columns id, dummy number column, score, dummy score, text
    returns the score and the text
    """
    tid, e_set, score, score2, text = [], [], [], [], []
    combined_raw = open(filename).read()
    raw_lines = combined_raw.splitlines()
    for row in range(1, len(raw_lines)):
        tid1, set1, score1, score12, text1 = raw_lines[row].strip().split("\t")
        tid.append(int(tid1))
        text.append(text1)
        e_set.append(int(set1))
        score.append(int(score1))
        score2.append(int(score12))

    return score, text


def read_in_test_prompt(filename):
    """
    Reads in the prompt from a text file
    Returns string
    """
    prompt_string = open(filename).read()
    return prompt_string

def read_in_test_data_twocolumn(filename,sep=","):
    """
    Reads in a two column version of the test data.
    Filename must point to a delimited file.
    In filename, the first column should be integer score data.
    The second column should be string text data.
    Sep specifies the type of separator between fields.
    """
    score, text = [], []
    combined_raw = open(filename).read()
    raw_lines = combined_raw.splitlines()
    for row in range(1, len(raw_lines)):
        score1, text1 = raw_lines[row].strip().split("\t")
        text.append(text1)
        score.append(int(score1))

    return score, text


def create_essay_set(text, d_score, n_score, p_score, prompt_string, generate_additional=True):
    """
    Creates an essay set from given data.
    Text should be a list of strings corresponding to essay text.
    Score should be a list of scores where score[n] corresponds to text[n]
    Prompt string is just a string containing the essay prompt.
    Generate_additional indicates whether to generate additional essays at the minimum score point or not.
    """
    x = EssaySet()
    for i in range(0, len(text)):
        print(i)

        x.add_essay(text[i], d_score[i], n_score[i], p_score[i]) #데이터셋 Essay을 전처리하여 추가한다.

        # 데이터 증강
        if d_score[i] == min(d_score) or n_score[i] == min(n_score) or p_score[i] == min(p_score) and generate_additional == True:
            x.generate_additional_essays(x._clean_text[len(x._clean_text) - 1], d_score[i], n_score[i], p_score[i])

    x.update_prompt(prompt_string)

    return x

def get_cv_error(clf,feats,scores):
    """
    Gets cross validated error for a given classifier, set of features, and scores
    clf - classifier
    feats - features to feed into the classified and cross validate over
    scores - scores associated with the features -- feature row 1 associates with score 1, etc.
    """
    results={'success' : False, 'kappa' : 0, 'mae' : 0}
    try:
        # cv_preds=util_functions.gen_cv_preds(clf,feats,scores)
        cv_preds=util_functions.gen_preds2(clf, feats, scores)
        err=numpy.mean(numpy.abs(numpy.array(cv_preds)-scores))
        kappa=util_functions.quadratic_weighted_kappa(list(cv_preds),scores)
        results['mae']=err
        results['kappa']=kappa
        results['success']=True
    except ValueError as ex:
        # If this is hit, everything is fine.  It is hard to explain why the error occurs, but it isn't a big deal.
        msg = u"Not enough classes (0,1,etc) in each cross validation fold: {ex}".format(ex=ex)
        log.debug(msg)
    except Exception as e:
        print(e)
        log.exception("Error getting cv error estimates.")

    return results

def get_algorithms(algorithm, model_type=False):
    """
    Gets two classifiers for each type of algorithm, and returns them.  First for predicting, second for cv error.
    type - one of util_functions.AlgorithmTypes
    """
    if algorithm == util_functions.AlgorithmTypes.classification:
        if model_type == 'lgbm':
            clf = lightgbm.LGBMClassifier(n_estimators=400) #, device_type='gpu')
            clf2 = lightgbm.LGBMClassifier(n_estimators=400) #, device_type='gpu')
        elif model_type == 'xgboost':
            clf = XGBClassifier(n_estimators=100, learning_rate=.01, max_depth=3, colsample_bytree = 0.8, subsample= 0.8, reg_lambda = 1, reg_alpha= 1)
            clf2 = XGBClassifier(n_estimators=100, learning_rate=.01, max_depth=3, colsample_bytree = 0.8, subsample= 0.8, reg_lambda = 1,  reg_alpha = 1)
        else:
            clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=.05,
                max_depth=4, random_state=1,min_samples_leaf=3)
            clf2=sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=.05,
                max_depth=4, random_state=1,min_samples_leaf=3)
    else:
        if model_type == 'lgbm':
            clf = lightgbm.LGBMRegressor(n_estimators=400) #, device_type='gpu')
            clf2 = lightgbm.LGBMRegressor(n_estimators=400) #, device_type='gpu')
        elif model_type == 'xgboost':
            clf = XGBRegressor(n_estimators=100, learning_rate=.05, max_depth=4)

            clf2 = XGBRegressor(n_estimators=100, learning_rate=.05, max_depth=4)
        else:
            clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=.05,
                max_depth=4, random_state=1,min_samples_leaf=3)
            clf2=sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=.05,
                max_depth=4, random_state=1,min_samples_leaf=3)
    return clf, clf2


def extract_features_and_generate_model_predictors(predictor_set, algorithm=util_functions.AlgorithmTypes.regression):
    """
    Extracts features and generates predictors based on a given predictor set
    predictor_set - a PredictorSet object that has been initialized with data
    type - one of util_functions.AlgorithmType
    """
    if(algorithm not in [util_functions.AlgorithmTypes.regression, util_functions.AlgorithmTypes.classification]):
        algorithm = util_functions.AlgorithmTypes.regression

    f = predictor_extractor.PredictorExtractor()
    f.initialize_dictionaries(predictor_set)

    train_feats = f.gen_feats(predictor_set)

    clf,clf2 = get_algorithms(algorithm)
    cv_error_results=get_cv_error(clf2,train_feats,predictor_set._target)

    try:
        set_score = numpy.asarray(predictor_set._target, dtype=numpy.int)
        clf.fit(train_feats, set_score)
    except ValueError:
        log.exception("Not enough classes (0,1,etc) in sample.")
        set_score = predictor_set._target
        set_score[0]=1
        set_score[1]=0
        clf.fit(train_feats, set_score)

    return f, clf, cv_error_results


def extract_features_and_generate_model(essays, algorithm=util_functions.AlgorithmTypes.regression, model_type=False):
    """
    Feed in an essay set to get feature vector and classifier
    essays must be an essay set object
    additional array is an optional argument that can specify
    a numpy array of values to add in
    returns a trained FeatureExtractor object and a trained classifier
    """
    f = feature_extractor.FeatureExtractor()
    f.initialize_dictionaries(essays)

    if model_type == 'lgbm':
        feats, feats_name = f.gen_feats(essays, True)
        train_feats = pd.DataFrame(feats, columns=feats_name)
    elif model_type == 'xgboost':
        feats, feats_name = f.gen_feats(essays, True)
        train_feats = feats
    else:
        train_feats = f.gen_feats(essays)

    set_score = numpy.asarray(essays._score, dtype=numpy.int)
    if len(util_functions.f7(list(set_score)))>5:
        algorithm = util_functions.AlgorithmTypes.regression
    else:
        algorithm = util_functions.AlgorithmTypes.classification

    clf,clf2 = get_algorithms(algorithm, model_type=model_type)

    cv_error_results=get_cv_error(clf2,train_feats,essays._score)

    try:
        clf.fit(train_feats, set_score)
    except ValueError:
        log.exception("Not enough classes (0,1,etc) in sample.")
        set_score[0]=1
        set_score[1]=0
        clf.fit(train_feats, set_score)

    return f, clf, cv_error_results

def extract_features_and_generate_models(essays, algorithm=util_functions.AlgorithmTypes.regression, model_type=False):
    """
    Feed in an essay set to get feature vector and classifier
    essays must be an essay set object
    additional array is an optional argument that can specify
    a numpy array of values to add in
    returns a trained FeatureExtractor object and a trained classifier
    """
    f = feature_extractor.FeatureExtractor()
    f.initialize_dictionaries(essays)

    if model_type == 'lgbm':
        feats, feats_name = f.gen_feats(essays, True)
        train_feats = pd.DataFrame(feats, columns=feats_name)
    elif model_type == 'xgboost':
        feats, feats_name = f.gen_feats(essays, True)
        train_feats = feats
    else:
        train_feats = f.gen_feats(essays)

    set_d_score = numpy.asarray(essays._d_score, dtype=numpy.int)
    set_n_score = numpy.asarray(essays._n_score, dtype=numpy.int)
    set_p_score = numpy.asarray(essays._p_score, dtype=numpy.int)
    set_score = numpy.asarray(essays._score, dtype=numpy.int)
    if len(util_functions.f7(list(set_d_score)))>5:
        algorithm = util_functions.AlgorithmTypes.regression
    else:
        algorithm = util_functions.AlgorithmTypes.classification

    d_clf, d_clf2 = get_algorithms(algorithm, model_type=model_type)
    n_clf, n_clf2 = get_algorithms(algorithm, model_type=model_type)
    p_clf, p_clf2 = get_algorithms(algorithm, model_type=model_type)
    clf,clf2 = get_algorithms(algorithm, model_type=model_type)

    d_cv_error_results=get_cv_error(d_clf2,train_feats,essays._d_score)
    n_cv_error_results=get_cv_error(n_clf2,train_feats,essays._n_score)
    p_cv_error_results=get_cv_error(p_clf2,train_feats,essays._p_score)
    cv_error_results=get_cv_error(clf2,train_feats,essays._score)

    try:
        d_clf.fit(train_feats, set_d_score)
        n_clf.fit(train_feats, set_n_score)
        p_clf.fit(train_feats, set_p_score)
        # clf.fit(train_feats, set_score)
    except ValueError:
        log.exception("Not enough classes (0,1,etc) in sample.")
        set_score[0]=1
        set_score[1]=0
        clf.fit(train_feats, set_score)

    return f, d_clf, n_clf, p_clf, d_cv_error_results, n_cv_error_results, p_cv_error_results


def dump_model_to_file(prompt_string, feature_ext, classifier, text, score, model_path):
    """
    Writes out a models to a file.
    prompt string is a string containing the prompt
    feature_ext is a trained FeatureExtractor object
    classifier is a trained classifier
    model_path is the path of write out the models file to
    """
    model_file = {'prompt': prompt_string, 'extractor': feature_ext, 'models': classifier, 'text' : text, 'score' : score}
    pickle.dump(model_file, file=open(model_path, "w"))

def create_essay_set_and_dump_model(text,score,prompt,model_path,additional_array=None):
    """
    Function that creates essay set, extracts features, and writes out models
    See above functions for argument descriptions
    """
    essay_set=create_essay_set(text,score,prompt)
    feature_ext,clf=extract_features_and_generate_model(essay_set,additional_array)
    dump_model_to_file(prompt,feature_ext,clf,model_path)


