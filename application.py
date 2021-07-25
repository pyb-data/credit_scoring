from flask import Flask
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
application = Flask(__name__)
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer


from data_preparation import prepareData



@application.route('/')
def hello_world():

    from pickle import load, dump
    import joblib

    # Préparation des données
    dfApplication = prepareData('test')

    # Load du modèle
    #pipeline = joblib.load('pipeline.joblib')
    pipeline = load(open('pipeline.pkl','rb'))
    # Prédiction du score (Les colonnes du dataframe doivent être dans l'ordre attendu par le modèle)
    imp = load(open('feature_importance.pkl','rb'))   
    dfApplication['SCORE'] = pipeline.predict_proba(dfApplication[list(imp.feature)]).T[1]
    # On ordonne les colonnes selon leur importance
    cols = ['SK_ID_CURR','SCORE']
    cols.extend(list(imp.feature))
    dfApplication = dfApplication[cols]
    # Sauvegarde
    dump(list(dfApplication.columns), open('dfApplicationDash_col.pkl','wb'))
    dump(list(dfApplication.values), open('dfApplicationDash_value.pkl','wb'))     
    
    return 'Done!'
