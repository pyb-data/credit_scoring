# Librairies d'outils
import os
import datetime
import warnings
import urllib.request

# Librairies mathématiques
import random
import math
import scipy.stats as st
import statsmodels.api as sm

# Librairies de tableaux de données
import numpy as np
import pandas as pd

# Librairies graphiques
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
import seaborn as sns

from sklearn import decomposition
from sklearn import preprocessing


import datetime



import os
import datetime
import time
import pandas as pd
import warnings
from collections import Counter
import matplotlib.pyplot as plt
from myTools import *

from sklearn.model_selection import train_test_split

import sklearn
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from numpy import mean, std
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN




from sklearn.inspection import permutation_importance

from numpy import percentile

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 999
pd.set_option("max_colwidth", 200)



from pickle import dump
from pickle import load

from sklearn.metrics import r2_score

from sklearn.preprocessing import PowerTransformer



def shuffleDf(df1, seed=0):

    df = df1.copy()
    df = df.reset_index(drop=True)
    index_lst = df.index.to_list()
    random.seed(seed)
    random.shuffle(index_lst)
    df.index = index_lst
    df = df.reset_index(drop=False)
    df = df.sort_values("index")
    del df["index"] 
    return df




from sklearn.base import BaseEstimator, TransformerMixin
class IdentityTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1
    
    
def getPipeline(
        df,
        target='', 
        identifier='', 
        ordinalThreshold=100,
        defaultNumImputer=SimpleImputer(strategy='mean'), 
        defaultOrdImputer=SimpleImputer(strategy='most_frequent'), 
        defaultCatImputer=SimpleImputer(strategy='most_frequent'), 
        meanImputer=[], iterativeImputer=[],mostFrequentImputer=[], constantImputer={},
        power=[], quantile=[], kbins10=[],kbins50=[], kbins100=[],
        defaultScaler=MinMaxScaler(), 
        minmax=[], standard=[], robust=[], noScale=[],
        defaultEncoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-10),
        ordinal=[], onehot=[],
        over=None, under=None,
        model=DecisionTreeClassifier()):
        
    if (over, under) != (None, None):
        from imblearn.pipeline import Pipeline
    else:
        from sklearn.pipeline import Pipeline
    
    # Identification des variables
    feat_nunique = df.nunique()
    numerical_feat = [x for x in df.select_dtypes(include=['float64','int64']).columns if (x not in [identifier,target]) & (x in list(feat_nunique[feat_nunique >= ordinalThreshold].index))]
    ordinal_feat = [x for x in df.select_dtypes(include=['float64','int64']).columns if (x not in [identifier,target]) & (x in list(feat_nunique[feat_nunique < ordinalThreshold].index))]
    categorical_feat = [x for x in df.select_dtypes(include=['object', 'bool']).columns if x not in [identifier,target]]

    features = list(df.columns).copy()
    features.remove(identifier)
    features.remove(target)      
    
    # Construction d'un dictionnaire qui contient les transformations (imputer, transformer, scaler) à appliquer pour chaque variable
    dicoTransform = {}
    for feature in features:
        dicoTransform[feature] = {'imputer':np.nan, 'transformer': np.nan, 'scaler':np.nan}

    for feature in meanImputer:
        dicoTransform[feature]['imputer'] = SimpleImputer(strategy='mean')
    for feature in mostFrequentImputer:
        dicoTransform[feature]['imputer'] = SimpleImputer(strategy='most_frequent')
    for feature in iterativeImputer:
        dicoTransform[feature]['imputer'] = IterativeImputer()
    for feature, constant in constantImputer.items():
        dicoTransform[feature]['imputer'] = SimpleImputer(strategy='constant', fill_value=constant)
    for feature in power:
        dicoTransform[feature]['transformer'] = PowerTransformer(method='yeo-johnson')
    for feature in quantile:
        dicoTransform[feature]['transformer'] = QuantileTransformer()
    for feature in kbins10:
        dicoTransform[feature]['transformer'] = KBinsDiscretizer(n_bins=10, encode= 'ordinal' , strategy= 'uniform')
    for feature in kbins50:
        dicoTransform[feature]['transformer'] = KBinsDiscretizer(n_bins=50, encode= 'ordinal' , strategy= 'uniform')
    for feature in kbins100:
        dicoTransform[feature]['transformer'] = KBinsDiscretizer(n_bins=100, encode= 'ordinal' , strategy= 'uniform')
    for feature in ordinal:
        dicoTransform[feature]['transformer'] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-10)
    for feature in onehot:
        dicoTransform[feature]['transformer'] = OneHotEncoder(handle_unknown='ignore')
    for feature in minmax:
        dicoTransform[feature]['scaler'] = MinMaxScaler()
    for feature in standard:
        dicoTransform[feature]['scaler'] = StandardScaler()
    for feature in robust:
        dicoTransform[feature]['scaler'] = RobustScaler()
    for feature in noScale:
        dicoTransform[feature]['scaler'] = IdentityTransformer()
    
    for k, v in dicoTransform.items():
        
        if type(dicoTransform[k]['imputer']) == float:
            if k in numerical_feat:
                dicoTransform[k]['imputer'] = defaultNumImputer
            elif k in categorical_feat:
                dicoTransform[k]['imputer'] = defaultCatImputer
            elif k in ordinal_feat:
                dicoTransform[k]['imputer'] = defaultOrdImputer
            else:
                imputer = 1 / 0
        if type(dicoTransform[k]['transformer']) == float:
            if k in numerical_feat:
                dicoTransform[k]['transformer'] = IdentityTransformer()
            elif k in categorical_feat:
                dicoTransform[k]['transformer'] = defaultEncoder
            elif k in ordinal_feat:
                dicoTransform[k]['transformer'] = IdentityTransformer()
            else:
                transformer = 1 / 0
        if type(dicoTransform[k]['scaler']) == float:
            if k in numerical_feat:
                dicoTransform[k]['scaler'] = defaultScaler
            elif k in categorical_feat:
                dicoTransform[k]['scaler'] = IdentityTransformer()
            elif k in ordinal_feat:
                dicoTransform[k]['scaler'] = defaultScaler
            else:
                scaler = 1 / 0

    column_transform_list = []
    for k, v in dicoTransform.items():
        to_insert = True
        for col_tr in column_transform_list:
            if (type(dicoTransform[k]['imputer']) == type(col_tr[0])) & (type(dicoTransform[k]['transformer']) == type(col_tr[1])) & (type(dicoTransform[k]['scaler']) == type(col_tr[2])):
                to_insert = False
                if type(dicoTransform[k]['transformer']) == sklearn.preprocessing._discretization.KBinsDiscretizer:
                    if dicoTransform[k]['transformer'].n_bins != col_tr[1].n_bins:
                        to_insert = True       
                if type(dicoTransform[k]['imputer']) == sklearn.impute._base.SimpleImputer:
                    if dicoTransform[k]['imputer'].strategy != col_tr[0].strategy:
                        to_insert = True
                    elif (dicoTransform[k]['imputer'].strategy == 'constant') & (dicoTransform[k]['imputer'].fill_value != col_tr[0].fill_value):
                        to_insert = True
                if to_insert == False:
                    break
        if to_insert == True:
            column_transform_list.append([dicoTransform[k]['imputer'], dicoTransform[k]['transformer'], dicoTransform[k]['scaler']])

    i = 1
    transformers = []
    for steps_transform in column_transform_list:
        
        Transformer = Pipeline(steps=[('imp', steps_transform[0]),
                                      ('trans', steps_transform[1]),
                                      ('scal', steps_transform[2])])
        
        feature_list = []
        for k, v in dicoTransform.items():
            to_insert = False
            if (type(dicoTransform[k]['imputer']) == type(steps_transform[0])) & (type(dicoTransform[k]['transformer']) == type(steps_transform[1])) & (type(dicoTransform[k]['scaler']) == type(steps_transform[2])):
                to_insert = True
                if type(dicoTransform[k]['transformer']) == sklearn.preprocessing._discretization.KBinsDiscretizer:
                    if dicoTransform[k]['transformer'].n_bins != steps_transform[1].n_bins:
                        to_insert = False
                if type(dicoTransform[k]['imputer']) == sklearn.impute._base.SimpleImputer:
                    if dicoTransform[k]['imputer'].strategy != steps_transform[0].strategy:
                        to_insert = False
                    elif (dicoTransform[k]['imputer'].strategy == 'constant') & (dicoTransform[k]['imputer'].fill_value != steps_transform[0].fill_value):
                        to_insert = False
            if to_insert == True:
                feature_list.append(k)
                
        pipeline_step = (str(i), Transformer, feature_list)
        i += 1

        transformers.append(pipeline_step)


    preprocessor = ColumnTransformer(transformers=transformers)
    
    if (over, under) == (None, None):
        pipeline = Pipeline(steps=[('prep',preprocessor),  ('m', model)])
    elif (over != None) & (under == None):
        pipeline = Pipeline(steps=[('prep',preprocessor), ('over' , over),  ('m', model)])
    elif (over == None) & (under != None):
        pipeline = Pipeline(steps=[('prep',preprocessor), ('under' , under),  ('m', model)])
    else:
        pipeline = Pipeline(steps=[('prep',preprocessor), ('over' , over), ('under' , under),  ('m', model)])
    
    dump(pipeline, open('pipeline.pkl','wb')) 

    return pipeline







def prAndRocCurves(y_train, y_pred_train, y_test, y_pred_test, display_plot=False, title_plot = 'No title', dummy_strategy='constant', dummy_constant=1):

    # Calculating train AUC
    fpr_train, tpr_train, thr_train = roc_curve(y_train, y_pred_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_train)
    pr_auc_train = auc(recall_train, precision_train)

    # Calculating test AUC
    fpr_test, tpr_test, thr_test = roc_curve(y_test, y_pred_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    precision_test, recall_test, threshold_test = precision_recall_curve(y_test, y_pred_test)
    pr_auc_test = auc(recall_test, precision_test)

    
    if display_plot:
        
        size = 1
        nbPlot = 2
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
        
        sub = fig.add_subplot(1,nbPlot,1)
        sub.set_title('ROC curve')
        plt.plot(fpr_test,tpr_test, label = 'AUC test = %0.3f' % roc_auc_test, color='red')
        plt.plot(fpr_train,tpr_train, label = 'AUC train = %0.3f' % roc_auc_train, color='green')
        #plt.plot(fpr_dummy,tpr_dummy, color='black', label = 'AUC dummy = %0.2f' % roc_auc_dummy)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],linestyle='--')
        plt.axis('tight')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        sub = fig.add_subplot(1,nbPlot,2)
        sub.set_title('Precision-Recall curve')
        plt.plot(recall_test, precision_test, label = 'AUC test = %0.3f' % pr_auc_test, color='red')
        plt.plot(recall_train, precision_train, label = 'AUC train = %0.3f' % pr_auc_train, color='green')
        #plt.plot(recall_dummy, precision_dummy, color='black', label = 'AUC dummy = %0.2f' % pr_auc_dummy)
        plt.legend(loc = 'lower right')
        plt.axis('tight')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        
        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle(title_plot)
        
        plt.show()

    
        print('TRAIN SET: ROC AUC=%.3f, PR AUC=%.3f' % (roc_auc_train, pr_auc_train))
        print('TEST SET:  ROC AUC=%.3f, PR AUC=%.3f' % (roc_auc_test, pr_auc_test))        

    return [roc_auc_train, pr_auc_train, roc_auc_test, pr_auc_test]






def evaluateRocPrCurvesOnTrainTestSet(
                                        dfTrain,
                                        dfTest,
                                        target='', 
                                        identifier='', 
                                        ordinalThreshold=100,
                                        defaultNumImputer=SimpleImputer(strategy='mean'), 
                                        defaultOrdImputer=SimpleImputer(strategy='most_frequent'), 
                                        defaultCatImputer=SimpleImputer(strategy='most_frequent'), 
                                        meanImputer=[], iterativeImputer=[], mostFrequentImputer=[], constantImputer={},
                                        power=[], quantile=[], kbins10=[],kbins50=[], kbins100=[],
                                        defaultScaler=MinMaxScaler(), 
                                        minmax=[], standard=[], robust=[], noScale=[],
                                        defaultEncoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-10),
                                        ordinal=[], onehot=[],
                                        over=None, under=None,
                                        model=DecisionTreeClassifier(),
                                        display_plot=True,
                                        title_plot='No title'):
    

    # Pipeline de transformation et de modélisation
    pipeline = getPipeline(
                            dfTrain,
                            target=target, 
                            identifier=identifier, 
                            ordinalThreshold=ordinalThreshold,
                            defaultNumImputer=defaultNumImputer, 
                            defaultOrdImputer=defaultOrdImputer, 
                            defaultCatImputer=defaultCatImputer, 
                            meanImputer=meanImputer, iterativeImputer=iterativeImputer, mostFrequentImputer=mostFrequentImputer, constantImputer=constantImputer,
                            power=power, quantile=quantile, kbins10=kbins10, kbins50=kbins50, kbins100=kbins100,
                            defaultScaler=defaultScaler, 
                            minmax=minmax, standard=standard, robust=robust, noScale=noScale,
                            defaultEncoder=defaultEncoder,
                            ordinal=ordinal, onehot=onehot,
                            over=over, under=under,
                            model=model)

    features = list(dfTrain.columns).copy()
    features.remove(identifier)
    features.remove(target)          

    # Entrainement
    time1 = time.time()
    pipeline.fit(dfTrain[features], dfTrain[target])
    time2 = time.time()
    # Prédiction sur le train set
    time_train = time2 - time1
    y_pred_train = pipeline.predict_proba(dfTrain[features]).transpose()[1]
    time3 = time.time()
    time_pred = time3 - time2
    # Prédiction sur le test set
    y_pred_test = pipeline.predict_proba(dfTest[features]).transpose()[1]
    time4 = time.time()

    # Calcul des AUC ROC et PR et graphiques
    roc_auc_train, pr_auc_train, roc_auc_test, pr_auc_test = prAndRocCurves(dfTrain[target].values, y_pred_train, dfTest[target].values, y_pred_test, display_plot=display_plot, title_plot=title_plot)
    
    return roc_auc_train,pr_auc_train,roc_auc_test,pr_auc_test,time_train,time_pred,len(features)



# test ratio
def evaluateRocPrCurves(
                    df1,
                    dfTest=None,                
                    target='', 
                    identifier='', 
                    ordinalThreshold=100,
                    meanImputer=[], iterativeImputer=[], mostFrequentImputer=[], constantImputer={},
                    defaultNumImputer=SimpleImputer(strategy='mean'), 
                    defaultOrdImputer=SimpleImputer(strategy='most_frequent'), 
                    defaultCatImputer=SimpleImputer(strategy='most_frequent'), 
                    power=[], quantile=[], kbins10=[],kbins50=[], kbins100=[],
                    defaultScaler=MinMaxScaler(), 
                    minmax=[], standard=[], robust=[],  noScale=[],
                    defaultEncoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-10),
                    ordinal=[], onehot=[],
                    over=None, under=None,
                    model=DecisionTreeClassifier(), 
                    cv=10,
                    random_state=None,
                    seed=0,
                    test_size=0.2,
                    display_plot=True,
                    title_plot='No title'):


    df = df1.copy()
    
    # Listes des métriques
    roc_auc_train_list = []
    pr_auc_train_list = []
    roc_auc_test_list = []
    pr_auc_test_list = []
    time_train_list = []
    time_pred_list = []
    nb_feat_list = []
    nb_train_list = []
    title_plot_list = []
    
    # Si aucun random state n'est passé en paramètre alors on est en cross validation
    # sinon on est en train-test split avec le random state passé en paramètre
    if (random_state == None) & (dfTest == None):
        
        if type(seed) != list:
            seed = [seed]
        
        for s in np.arange(len(seed)):
            
            # On réordonne aléatoirement selon le seed passé en paramêtres
            df = df.sort_values(identifier, ascending=True)
            df = shuffleDf(df, seed=seed[s])
            df = df.reset_index(drop=True)
            # Construction des 10 sous sous-ensembles de cross-validation
            n = int(df.shape[0] / cv)
            lstDf = []
            for i in np.arange(cv):
                if i < cv-1:
                    lstDf.append(df.iloc[i*n:(i+1)*n,].copy())
                else:
                    lstDf.append(df.iloc[i*n:,].copy())        
            del df
            for i in np.arange(cv):
                dfTest = lstDf[i]
                lstTrainDf = []
                for j in np.arange(cv):
                    if j != i:
                        lstTrainDf.append(lstDf[j])
                dfTrain = pd.concat(lstTrainDf)  
                res = evaluateRocPrCurvesOnTrainTestSet(
                                    dfTrain,
                                    dfTest,
                                    target=target, 
                                    identifier=identifier, 
                                    ordinalThreshold=ordinalThreshold,
                                    defaultNumImputer=defaultNumImputer, 
                                    defaultOrdImputer=defaultOrdImputer, 
                                    defaultCatImputer=defaultCatImputer, 
                                    meanImputer=meanImputer, iterativeImputer=iterativeImputer, mostFrequentImputer=mostFrequentImputer, constantImputer=constantImputer,
                                    power=power, quantile=quantile, kbins10=kbins10,kbins50=kbins50, kbins100=kbins100,
                                    defaultScaler=defaultScaler, 
                                    minmax=minmax, 
                                    standard=standard, 
                                    robust=robust, 
                                    defaultEncoder=defaultEncoder,
                                    ordinal=ordinal, 
                                    onehot=onehot,over=over, under=under,
                                    model=model, 
                                    display_plot=display_plot,
                                    title_plot=title_plot)
                # Conservation des métriques
                roc_auc_train_list.append(res[0])
                pr_auc_train_list.append(res[1])
                roc_auc_test_list.append(res[2])
                pr_auc_test_list.append(res[3])
                time_train_list.append(res[4])
                time_pred_list.append(res[5])
                nb_feat_list.append(res[6])
                nb_train_list.append(dfTrain.shape[0])
                if len(seed) > 1:
                    title = title_plot + ' | '
                else:
                    title = title_plot
                title_plot_list.append(title)            

    else:
        cv = 1
        if random_state != None:
            # On fait un train-test split   
            if dfTest == None:     
                dfTrain, dfTest = train_test_split(df, test_size=test_size, random_state=random_state)        
            else:     
                dfTrain, _ = train_test_split(df, test_size=test_size, random_state=random_state)                
        else:
            # Ici il faut que dfTest soit passé en paramètre, sinon ça va planter    
            dfTrain = df
        res = evaluateRocPrCurvesOnTrainTestSet(
                            dfTrain,
                            dfTest,
                            target=target, 
                            identifier=identifier, 
                            defaultNumImputer=defaultNumImputer, 
                            defaultOrdImputer=defaultOrdImputer, 
                            defaultCatImputer=defaultCatImputer, 
                            ordinalThreshold=ordinalThreshold,
                            power=power, quantile=quantile, kbins10=kbins10,kbins50=kbins50, kbins100=kbins100,
                            defaultScaler=defaultScaler, 
                            minmax=minmax, 
                            standard=standard, 
                            robust=robust, 
                            defaultEncoder=defaultEncoder,
                            ordinal=ordinal, 
                            onehot=onehot,
                            over=over, under=under,
                            model=model, 
                            display_plot=display_plot,
                            title_plot=title_plot)        
        # Conservation des métriques
        roc_auc_train_list.append(res[0])
        pr_auc_train_list.append(res[1])
        roc_auc_test_list.append(res[2])
        pr_auc_test_list.append(res[3])
        time_train_list.append(res[4])
        time_pred_list.append(res[5])
        nb_feat_list.append(res[6])
        nb_train_list.append(dfTrain.shape[0])
        title_plot_list.append(title_plot)    
        

     
    try:
        del lstDf
    except:
        pass
    try:
        del df
    except:
        pass
    try:
        del dfTrain
        del dfTest
    except:
        pass
 

	    
    # SI plot, Affichage des métriques moyennes
    if (cv > 1) & (display_plot == True) & (random_state == None):
        # Calcul des métriques moyennes
        mean_roc_auc_train = np.array(roc_auc_train_list).mean()
        mean_pr_auc_train = np.array(pr_auc_train_list).mean()
        mean_roc_auc_test = np.array(roc_auc_test_list).mean()
        mean_pr_auc_test = np.array(pr_auc_test_list).mean()
        print('')
        print('_______________________________________________________________________________________________________________________________________')
        print('METRIQUES MOYENNES TEST:')
        print('ROC AUC = %.3f' % (mean_roc_auc_test))
        print('PR AUC  = %.3f' % (mean_pr_auc_test))
        print('_______________________________________________________________________________________________________________________________________')
        print('_______________________________________________________________________________________________________________________________________')
        print('METRIQUES MOYENNES TRAIN:')
        print('ROC AUC = %.3f' % (mean_roc_auc_train))
        print('PR AUC  = %.3f' % (mean_pr_auc_train))
        print('_______________________________________________________________________________________________________________________________________')        
        print('')
        print('')
    
    # Retour des métriques
    return pd.DataFrame({'title':title_plot_list,
                         'roc_auc_train':roc_auc_train_list,
                         'roc_auc_test':roc_auc_test_list,
                         'pr_auc_train':pr_auc_train_list,
                         'pr_auc_test':pr_auc_test_list,
                         'time_train':time_train_list,
                         'time_pred':time_pred_list,
                         'nb_feat':nb_feat_list,
                         'nb_train':nb_train_list,
                         'timestamp': [round(time.time(),0) for x in title_plot_list]
                        })





def q75(serie):
    return serie.quantile(q=0.75)


def q25(serie):
    return serie.quantile(q=0.25)


def displayPlotParamOptim(results, show_quantile=True):

    dfMetrics = pd.concat(results)
    dfMetrics = dfMetrics.rename(columns={'pr_train_auc': 'pr_auc_train'})
    dfMetrics['model'] = dfMetrics.apply(lambda x: x.title.split('|')[0], axis=1)
    dfMetrics['parameter'] = dfMetrics.apply(lambda x: x.title.split('|')[-1].split('=')[0], axis=1)
    dfMetrics['value'] = dfMetrics.apply(lambda x: float(x.title.split('|')[-1].split('=')[1]), axis=1)

    dfModelParam = dfMetrics[['model','parameter']].drop_duplicates()

    for index, row in dfModelParam.iterrows():

        m = dfMetrics[(dfMetrics.model == row.model) & (dfMetrics.parameter == row.parameter)]
        m_means = m.groupby('value').agg({'roc_auc_train':'mean','pr_auc_train':'mean','roc_auc_test':'mean','pr_auc_test':'mean','time_train':'mean','time_pred':'mean'}).reset_index(drop=False)
        if show_quantile == True:
            m_q75 = m.groupby('value').agg({'roc_auc_train':q75,'pr_auc_train':q75,'roc_auc_test':q75,'pr_auc_test':q75,'time_train':q75,'time_pred':q75}).reset_index(drop=False)
            m_q25 = m.groupby('value').agg({'roc_auc_train':q25,'pr_auc_train':q25,'roc_auc_test':q25,'pr_auc_test':q25,'time_train':q25,'time_pred':q25}).reset_index(drop=False)

        size = 1
        nbPlot = 3
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))

        sub = fig.add_subplot(1,nbPlot,1)
        plt.xlabel(row.parameter)
        plt.ylabel('AUC')
        plt.title('ROC AUC')
        #plt.axis('tight')
        plt.grid(True)
        plt.scatter(m.value.values, m.roc_auc_train.values, alpha=1, label = 'train')
        if show_quantile == True:
            plt.plot(m_means.value, m_means.roc_auc_train, color='blue')
            #plt.plot(m_q75.value, m_q75.roc_auc_train, color='grey')
            #plt.plot(m_q25.value, m_q25.roc_auc_train, color='grey')
        else:
            plt.plot(m_means.value, m_means.roc_auc_train, color='grey')
        plt.scatter(m.value.values, m.roc_auc_test.values, alpha=1, label = 'test')
        if show_quantile == True:
            plt.plot(m_means.value, m_means.roc_auc_test, color='orange')
            plt.plot(m_q75.value, m_q75.roc_auc_test, color='orange', linestyle='dotted')
            plt.plot(m_q25.value, m_q25.roc_auc_test, color='orange', linestyle='dotted')
        else:
            plt.plot(m_means.value, m_means.roc_auc_test, color='grey')
        plt.legend()
        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle('ROC AUC')

        sub = fig.add_subplot(1,nbPlot,2)
        plt.xlabel(row.parameter)
        plt.ylabel('AUC')
        plt.title('PR AUC')
        #plt.axis('tight')
        plt.grid(True)
        plt.scatter(m.value.values, m.pr_auc_train.values, alpha=1, label = 'train')
        if show_quantile == True:
            plt.plot(m_means.value, m_means.pr_auc_train, color='blue')
            #plt.plot(m_q75.value, m_q75.pr_auc_train, color='grey')
            #plt.plot(m_q25.value, m_q25.pr_auc_train, color='grey')     
        else:
            plt.plot(m_means.value, m_means.pr_auc_train, color='grey')
        plt.scatter(m.value.values, m.pr_auc_test.values, alpha=1, label = 'test')
        if show_quantile == True:
            plt.plot(m_means.value, m_means.pr_auc_test, color='orange')
            plt.plot(m_q75.value, m_q75.pr_auc_test, color='orange', linestyle='dotted')
            plt.plot(m_q25.value, m_q25.pr_auc_test, color='orange', linestyle='dotted') 
        else:
            plt.plot(m_means.value, m_means.pr_auc_test, color='grey')
        plt.legend()
        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle('PR AUC')
        
        sub = fig.add_subplot(1,nbPlot,3)
        plt.xlabel(row.parameter)
        plt.ylabel('duration')
        plt.title('Training and predicting duration')
        #plt.axis('tight')
        plt.grid(True)
        plt.scatter(m.value.values, m.time_train.values, alpha=1, label = 'training')
        plt.plot(m_means.value, m_means.time_train, color='blue')
        plt.scatter(m.value.values, m.time_pred.values, alpha=1, label = 'predicting')
        plt.plot(m_means.value, m_means.time_pred, color='orange')
        plt.legend()
        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle('Training and predicting duration')

        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle("{} - {}".format(row.model, row.parameter))

        plt.show()






def displayPlotParamOptimBox(results):

    dfMetrics = pd.concat(results)
    dfMetrics = dfMetrics.rename(columns={'pr_train_auc': 'pr_auc_train'})
    dfMetrics['model'] = dfMetrics.apply(lambda x: x.title.split('|')[0], axis=1)
    dfMetrics['parameter'] = dfMetrics.apply(lambda x: x.title.split('|')[-1].split('=')[0], axis=1)
    dfMetrics['value'] = dfMetrics.apply(lambda x: x.title.split('|')[-1].split('=')[1], axis=1)

    dfModelParam = dfMetrics[['model','parameter']].drop_duplicates()

    for index, row in dfModelParam.iterrows():

        x_labels = []
        y_roc_auc_train = []
        y_roc_auc_test = []
        y_pr_auc_train = []
        y_pr_auc_test = []
        y_time_train = []
        y_time_pred = []
        
        m = dfMetrics[(dfMetrics.model == row.model) & (dfMetrics.parameter == row.parameter)]

        for val in m['value'].unique():
            x_labels.append(val)
            y_roc_auc_train.append(m[m['value']==val].roc_auc_train)
            y_roc_auc_test.append(m[m['value']==val].roc_auc_test)
            y_pr_auc_train.append(m[m['value']==val].pr_auc_train)
            y_pr_auc_test.append(m[m['value']==val].pr_auc_test)
            y_time_train.append(m[m['value']==val].time_train)
            y_time_pred.append(m[m['value']==val].time_pred)

        size = 1
        nbPlot = 3
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))

        sub = fig.add_subplot(1,nbPlot,1)
        plt.xlabel(row.parameter)
        plt.ylabel('AUC')
        plt.title('ROC AUC')
        #plt.axis('tight')
        plt.grid(True)
        plt.boxplot(y_roc_auc_train, labels=x_labels, showmeans=True)
        plt.boxplot(y_roc_auc_test, labels=x_labels, showmeans=True)
        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle('ROC AUC')

        sub = fig.add_subplot(1,nbPlot,2)
        plt.xlabel(row.parameter)
        plt.ylabel('AUC')
        plt.title('PR AUC')
        #plt.axis('tight')
        plt.grid(True)
        plt.boxplot(y_pr_auc_train, labels=x_labels, showmeans=True)
        plt.boxplot(y_pr_auc_test, labels=x_labels, showmeans=True)
        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle('PR AUC')
        
        sub = fig.add_subplot(1,nbPlot,3)
        plt.xlabel(row.parameter)
        plt.ylabel('duration')
        plt.title('Training and predicting duration')
        #plt.axis('tight')
        plt.grid(True)
        plt.boxplot(y_time_train, labels=x_labels, showmeans=True)
        plt.boxplot(y_time_pred, labels=x_labels, showmeans=True)
        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle('Training and predicting duration')

        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle("{} - {}".format(row.model, row.parameter))

        plt.show()



def ColMode(df, feature, key):
    
    if type(key) == str:
        key = [key]
    if type(feature) == str:
        feature = [feature]
    
    # dataframe de travail sans les nans et avec les colonnes feature et key
    cols = feature.copy()
    cols.extend(key)
    tmp = df[cols].dropna()
    
    # Count par feature-key
    tmp = pd.DataFrame(tmp.groupby(cols).size(), columns=['COUNT']).reset_index(drop=False)
    
    # Par key, valeurs les plus fréquentes de la feature
    tmpMaxCount = tmp.groupby(by=key).agg({'COUNT': 'max'}).reset_index(drop=False)
    colsMerge = key.copy()
    colsMerge.append('COUNT')
    tmp = tmp.merge(tmpMaxCount, left_on=colsMerge, right_on=colsMerge)

    # A chaque valeur de feature on affecte un poids égal au nombre de fois que la valeur est prise dans la table
    colsPopularity = feature.copy()
    colsPopularity.append('COUNT')
    popularity = tmp.groupby(feature).agg({'COUNT': 'sum'}).reset_index(drop=False)
    colsPopularity = feature.copy()
    colsPopularity.append('POPULARITY')
    popularity.columns = colsPopularity
    
    tmp = tmp.merge(popularity)
    
    tmp2 = tmp.groupby(key).agg({'COUNT': 'max'}).reset_index(drop=False)
    tmp = tmp.merge(tmp2)
    tmp2 = tmp.groupby(key).agg({'POPULARITY': 'min'}).reset_index(drop=False)
    tmp = tmp.merge(tmp2)
    
    del tmp['COUNT']
    del tmp['POPULARITY']
    
    return tmp

def TransformUnique(df, key):
    categorical_ix = df.select_dtypes(include=['object']).columns
    dfCat = df[key].drop_duplicates()
    for col in categorical_ix:
        try:
            dfCat = dfCat.merge(ColMode(df, col, key), left_on=key, right_on=key, how='left')
        except:
            print(col)
    dfNum = df.groupby(key).mean().reset_index(drop=False)
    dfUnique = dfCat.merge(dfNum, left_on=key, right_on=key, how='left')
    return dfUnique




def statsMissingValues(df):
    rateOfMissing = (df.isnull()).sum() / df.shape[0] * 100
    rateOfMissing = rateOfMissing.reset_index(drop=False)
    rateOfMissing.columns = ['feature','missingRate']
    #return rateOfMissing
    return rateOfMissing[rateOfMissing.missingRate > 0].sort_values('missingRate', ascending=False)



def delColWIthMissing(df, missingRateThreshold=1):
    missingRates = (df.isnull()).sum() / df.shape[0]
    for col in missingRates.index:
        if missingRates[col] >= missingRateThreshold:
            del df[col]
    return df





from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def featureImportance(df1, target, identifier, nbCalc=3, model=DecisionTreeClassifier()):

    df = df1.copy()
    
    y = df[target]
    del df[target]
    del df[identifier]
            
    for i in np.arange(nbCalc):
        model.fit(df, y)
        try:
            imp = model.feature_importances_
        except:
            imp = model.named_steps['m'].feature_importances_
        if i == 0:
            importance = pd.DataFrame(imp)
        else:
            importance = pd.concat([importance,pd.DataFrame(imp)], axis=1)
        i += 1
        
    statImp = pd.DataFrame(list(df.columns), columns=['feature'])
    
    statImp['impMin'] = np.min(importance, axis=1)
    statImp['impQ25'] = np.quantile(importance, 0.25, axis=1)
    statImp['impMean'] = np.mean(importance, axis=1)
    statImp['impMedian'] = np.median(importance, axis=1)
    statImp['impQ75'] = np.quantile(importance, 0.75, axis=1)
    statImp['impMax'] = np.max(importance, axis=1)
    
    statImp = statImp.sort_values('impMedian', ascending=False).reset_index(drop=True).reset_index(drop=False)
    statImp = statImp.rename(columns={'index':'feat_imp_classement'})
    statImp['feat_imp_classement'] = statImp.apply(lambda x: x.feat_imp_classement+1, axis=1)
    
    return statImp


def permutationImportance(df1, target, identifier, model=DecisionTreeClassifier(), scoring='roc_auc'):
    
    y, X, X_col = getyAndX(df1, target, identifier)
    results = permutation_importance(model, X, y, scoring=scoring)
    permImp = pd.DataFrame(results.importances_mean, columns=['perm_imp'])
    permImp['feature'] = X_col
    permImp = permImp[['feature','perm_imp']]
    
    permImp = permImp.sort_values('perm_imp', ascending=False).reset_index(drop=True).reset_index(drop=False)
    permImp = permImp.rename(columns={'index':'perm_classement'})
    permImp['perm_classement'] = permImp.apply(lambda x: x.perm_classement+1, axis=1)
    
    return permImp


def featureAndPermutationImportance(df1, target, identifier, model=DecisionTreeClassifier(), scoring='roc_auc'):
    perm_imp = permutationImportance(X, y, X_col, model, scoring=scoring)
    imp = featureImportance(X, y, X_col, model)
    imp = imp.merge(perm_imp, left_on='feature', right_on='feature')
    imp['diff_classement'] = imp.apply(lambda x: np.abs(x.feat_imp_classement - x.perm_classement), axis=1)
    imp['best_classement'] = imp.apply(lambda x: np.min([x.feat_imp_classement, x.perm_classement]), axis=1)
    imp = imp[['feature', 'impMean', 'impMin', 'impQ25',  'impMedian', 'impQ75', 'impMax', 'perm_imp', 'feat_imp_classement', 'perm_classement', 'diff_classement', 'best_classement']]
    
    size = 1
    nbPlot = 2
    fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
    
    sub = fig.add_subplot(1,nbPlot,1)
    sub.set_title('Feature importance')
    plt.bar([x for x in range(len(imp))], imp['impMedian'].sort_values(ascending=False))
    plt.axis('tight')
    plt.xlabel('Features')
    plt.ylabel('Importance')

    sub = fig.add_subplot(1,nbPlot,2)
    sub.set_title('Permutation importance')
    plt.bar([x for x in range(len(imp))], imp['perm_imp'].sort_values(ascending=False))
    plt.axis('tight')
    plt.xlabel('Features')
    plt.ylabel('Importance')

    plt.rcParams.update({'font.size':12, 'font.style':'normal'})
    plt.suptitle("Feature importance")

    plt.show()
    
    return imp


def evaluateFeatureImportance(df1, 
                                target='', 
                                identifier='', 
                                ordinalThreshold=100,
                                defaultNumImputer=SimpleImputer(strategy='mean'), 
                                defaultOrdImputer=SimpleImputer(strategy='most_frequent'), 
                                defaultCatImputer=SimpleImputer(strategy='constant', fill_value='missing'), 
                                meanImputer=[], iterativeImputer=[], mostFrequentImputer=[], constantImputer={},
                                power=[], quantile=[], kbins10=[],kbins50=[], kbins100=[],
                                defaultScaler=MinMaxScaler(), 
                                minmax=[], standard=[], robust=[], noScale=[],
                                defaultEncoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-10),
                                ordinal=[], onehot=[],
                                models=[RandomForestClassifier(), XGBClassifier()],
                                label_models=['Rando forest','XGBoost'],
                                show_plot=True,                              
                                title_plot='No title'):

    df = df1.copy()
    
    pipeline1 = getPipeline(
                        df,
                        target=target, 
                        identifier=identifier, 
                        ordinalThreshold=ordinalThreshold,
                        defaultNumImputer=defaultNumImputer, 
                        defaultOrdImputer=defaultOrdImputer, 
                        defaultCatImputer=defaultCatImputer, 
                        meanImputer=meanImputer, iterativeImputer=iterativeImputer,mostFrequentImputer=mostFrequentImputer, constantImputer=constantImputer,
                        power=power, quantile=quantile, kbins10=kbins10,kbins50=kbins50, kbins100=kbins100,
                        defaultScaler=defaultScaler, 
                        minmax=minmax, standard=standard, robust=robust, noScale=noScale,
                        defaultEncoder=defaultEncoder,
                        ordinal=ordinal, onehot=onehot,
                        model=models[0])
    
    pipeline2 = getPipeline(
                        df,
                        target=target, 
                        identifier=identifier, 
                        ordinalThreshold=ordinalThreshold,
                        defaultNumImputer=defaultNumImputer, 
                        defaultOrdImputer=defaultOrdImputer, 
                        defaultCatImputer=defaultCatImputer, 
                        meanImputer=meanImputer, iterativeImputer=iterativeImputer,mostFrequentImputer=mostFrequentImputer, constantImputer=constantImputer,
                        power=power, quantile=quantile, kbins10=kbins10,kbins50=kbins50, kbins100=kbins100,
                        defaultScaler=defaultScaler, 
                        minmax=minmax, standard=standard, robust=robust, noScale=noScale,
                        defaultEncoder=defaultEncoder,
                        ordinal=ordinal, onehot=onehot,
                        model=models[1])
    
    impRF = featureImportance(df, identifier=identifier, target=target, model=pipeline1)
    impRF = impRF[['feature','impMedian','feat_imp_classement']].rename(columns={'impMedian':'RFFeatImp','feat_imp_classement':'RFClassement'})
    impXG = featureImportance(df, identifier=identifier, target=target, model=pipeline2)
    impXG = impXG[['feature','impMedian','feat_imp_classement']].rename(columns={'impMedian':'XGFeatImp','feat_imp_classement':'XGClassement'})
    imp = impRF.merge(impXG, left_on='feature', right_on='feature')
    
    imp['bestClassement'] = imp.apply(lambda x: np.min([x.RFClassement, x.XGClassement]), axis=1)
    imp['diffClassement'] = imp.apply(lambda x: np.abs(x.RFClassement - x.XGClassement), axis=1)
    
    imp = imp[['feature','RFFeatImp','XGFeatImp','RFClassement','XGClassement','bestClassement','diffClassement']]
    imp = imp.sort_values('bestClassement')

    if show_plot:
        
        size = 1
        nbPlot = 2
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))

        sub = fig.add_subplot(1,nbPlot,1)
        sub.set_title(label_models[0])
        plt.bar([x for x in range(len(imp))], imp['RFFeatImp'].sort_values(ascending=False))
        plt.axis('tight')
        plt.xlabel('Features')
        plt.ylabel('Importance')

        sub = fig.add_subplot(1,nbPlot,2)
        sub.set_title(label_models[1])
        plt.bar([x for x in range(len(imp))], imp['XGFeatImp'].sort_values(ascending=False))
        plt.axis('tight')
        plt.xlabel('Features')
        plt.ylabel('Importance')

        plt.rcParams.update({'font.size':12, 'font.style':'normal'})
        plt.suptitle(title_plot)


        plt.show()
        
    return imp.sort_values('bestClassement')






def stats(df1, identifier='', target='', inclCol=[], whis=1.5, excludeDominantExtrem=True):
    
    if type(target) == str:
        target = [target]
    if type(identifier) == str:
        identifier = [identifier]
    
    colToExclude = target
    colToExclude.extend(identifier)
    
    numerical_ix = list(df1.select_dtypes(include=['float64','int64']).columns)
    dico_nunique = {}
    dico_nb_not_missing = {}
    dico_missing_prct = {}
    dico_nb_outliers = {}
    dico_cut_off_up = {}
    dico_nb_outliers_up = {}
    dico_cut_off_down = {}
    dico_nb_outliers_down = {}
    dico_min = {}
    dico_q25 = {}
    dico_median = {}
    dico_q75 = {}
    dico_max = {}
    dico_skew = {}
    dico_kurt = {}
    dico_mean = {}
    dico_std = {}
    dico_norm_range_left = {}
    dico_norm_range_center = {}
    dico_norm_range_right = {}
    dico_norm_range_corr = {}

    nb_tot_rec = df1.shape[0]
    
    for col in numerical_ix:
        
        if col not in colToExclude:
            
            if (col in inclCol) or (inclCol == []):
            
                df = df1.copy()
                df = df[~df[col].isnull()]
                dataCol = df[col]
                nb_not_missing = len(dataCol)
                missing_prct = (nb_tot_rec - nb_not_missing) / nb_tot_rec * 100

                min_val = np.min(dataCol)
                max_val = np.max(dataCol)
                q25 = percentile(dataCol, 25)
                q75 = percentile(dataCol, 75) 
                lower = -999
                upper = 999            
                if excludeDominantExtrem and (dataCol.nunique() > 1000):
                    if q25 == min_val:
                        lower = min_val
                        dataCol = df[df[col] > min_val][col]
                        q25, q75 = percentile(dataCol, 25), percentile(dataCol, 75)
                    if q75 == max_val:
                        upper = max_val
                        dataCol = df[df[col] < max_val][col]
                        q25, q75 = percentile(dataCol, 25), percentile(dataCol, 75)
                iqr = q75 - q25
                median = np.median(dataCol)
                mean = np.mean(dataCol)
                std = np.std(dataCol)
                # see for whis:
                # https://stackoverflow.com/questions/17725927/boxplots-in-matplotlib-markers-and-outliers
                cut_off = iqr * whis
                if lower == -999:
                    lower = q25 - cut_off            
                if upper == 999:
                    upper = q75 + cut_off
                nb_outliers_down = df[df[col] < lower].shape[0] 
                nb_outliers_up = df[df[col] > upper].shape[0]
                nb_outliers = nb_outliers_up + nb_outliers_down

                dico_nunique[col] = dataCol.nunique()
                dico_nb_not_missing[col] = nb_not_missing
                dico_missing_prct[col] = missing_prct

                dico_q25[col] = q25
                dico_min[col] = min_val      
                dico_median[col] = median     
                dico_q75[col] = q75
                dico_max[col] = max_val

                dico_mean[col] = mean
                dico_std[col] = std

                #dico_nb_outliers[col] = nb_outliers
                dico_cut_off_up[col] = upper
                dico_nb_outliers_up[col] = nb_outliers_up
                dico_cut_off_down[col] = lower
                dico_nb_outliers_down[col] = nb_outliers_down

                dico_skew[col] = st.skew(dataCol)
                dico_kurt[col] = st.kurtosis(dataCol)

                #n1 = df[(df[col]>=mean-3*std) & (df[col]<=mean-std)].shape[0]/nb_not_missing
                #n2 = df[(df[col]>=mean-std) & (df[col]<=mean+std)].shape[0]/nb_not_missing
                #n3 = df[(df[col]>=mean+std) & (df[col]<=mean+3*std)].shape[0]/nb_not_missing
                #dico_norm_range_left[col] = n1
                #dico_norm_range_center[col] = n2
                #dico_norm_range_right[col] = n3
                #dico_norm_range_corr[col] = r2_score([n1,n2,n3],[0.1573,0.6827,0.1573])
            
    del df
            
    res = pd.DataFrame({'feature': list(dico_nunique.keys()),
                        'nb_unique': list(dico_nunique.values()),
                        'missing_prct': list(dico_missing_prct.values()),
                        'mini': list(dico_min.values()),
                        'q25': list(dico_q25.values()),
                        'median': list(dico_median.values()),
                        'q75': list(dico_q75.values()),
                        'maxi': list(dico_max.values()),
                        'mean': list(dico_mean.values()),
                        'std': list(dico_std.values()),
                        'skewness': list(dico_skew.values()),
                        'kurtosis': list(dico_kurt.values()), 
                        #'norm_range_corr': list(dico_norm_range_corr.values()),                        
                        #'norm_range_left': list(dico_norm_range_left.values()),
                        #'norm_range_center': list(dico_norm_range_center.values()),
                        #'norm_range_right': list(dico_norm_range_right.values()),
                        #'nb_not_missing': list(dico_nb_not_missing.values()),
                        #'nb_outliers': list(dico_nb_outliers.values()),
                        'cut_off_down': list(dico_cut_off_down.values()),                        
                        'nb_outliers_down': list(dico_nb_outliers_down.values()),
                        'cut_off_up': list(dico_cut_off_up.values()),                        
                        'nb_outliers_up': list(dico_nb_outliers_up.values())
                        
                       })
    
    return res





def distribRatio(df1, feature, target, bins=100):
    
    df = df1.copy()

    mini = df[feature].min()
    maxi = df[feature].max()

    l_inf = [mini + x * (maxi-mini)/bins for x in np.arange(bins)]
    l_sup = [mini + (x+1) * (maxi-mini)/bins for x in np.arange(bins)]

    res_inf = []
    res_inf_incl = []
    res_sup = []
    res_sup_incl = []
    nb = []
    ratio = []

    tmp = df[df[feature] == mini].copy()
    if tmp.shape[0] > df.shape[0] / 100:
        bins_mini = True
    else:
        bins_mini = False

    tmp = df[df[feature] == maxi].copy()
    if tmp.shape[0] > df.shape[0] / 100:
        bins_maxi = True
    else:
        bins_maxi = False

    tmp = df[(df[feature] >= l_inf[bins-1]) & (df[feature] <= l_inf[bins-1])].copy()
    if tmp.shape[0] < 1000:
        last_bin_less_that_1000 = True
    else:
        last_bin_less_that_1000 = False   

    i=1
    last_loop = False
    keepPreviousInf = False
    for interval in zip(l_inf, l_sup):
        if keepPreviousInf == False:
            borne_inf = interval[0]

        if last_bin_less_that_1000 and (i == bins-1):
            borne_sup = maxi
        elif last_bin_less_that_1000 & (i == bins):
            break
        else:
            borne_sup = interval[1]

        if (borne_inf == mini) & (bins_mini == True):
            # On fait un bins spécial pour le mini
            tmp = df[df[feature]==mini].copy()
            nb_pos = tmp[(tmp[target]==1)].shape[0]
            nb_tot = tmp.shape[0]
            res_inf.append(mini)
            res_inf_incl.append(True)
            res_sup.append(mini)
            res_sup_incl.append(True)   
            nb.append(nb_tot)
            ratio.append(nb_pos / nb_tot)
            # Et pour le suivant on exlu le mini
            tmp = df[(df[feature]>borne_inf) & (df[feature]<borne_sup)].copy()
            nb_pos = tmp[(tmp[target]==1)].shape[0]
            nb_tot = tmp.shape[0]
            if nb_tot < 1000:
                keepPreviousInf = True
            else:
                keepPreviousInf = False
                res_inf.append(borne_inf)
                res_inf_incl.append(False)
                res_sup.append(borne_sup)
                res_sup_incl.append(False)  
                nb.append(nb_tot)
                ratio.append(nb_pos / nb_tot)
        else:    
            if (i == bins) | (last_bin_less_that_1000 and (i == bins-1)):     
                tmp = df[(df[feature]>=borne_inf) & (df[feature]<=borne_sup)].copy()    

            else:                 
                tmp = df[(df[feature]>=borne_inf) & (df[feature]<borne_sup)].copy()
            nb_pos = tmp[(tmp[target]==1)].shape[0]
            nb_tot = tmp.shape[0]
            if (nb_tot < 1000) and (i < bins) and not (last_bin_less_that_1000 and (i == bins-1)):
                keepPreviousInf = True
            else:
                keepPreviousInf = False
                res_inf.append(borne_inf)
                res_sup.append(borne_sup)                     
                if (i == bins) | (last_bin_less_that_1000 and (i == bins-1)):   
                    res_inf_incl.append(True)            
                    res_sup_incl.append(True)                   
                else:
                    res_inf_incl.append(True)            
                    res_sup_incl.append(False) 
                nb.append(nb_tot)
                if (i == bins) & (nb_tot < 1000):
                    print("il ne faut pas passer par là!")
                    ratio.append(previous_ratio)
                else:
                    ratio.append(nb_pos / nb_tot)
                previous_ratio = nb_pos / nb_tot

        
        i+=1

    return pd.DataFrame({'feature': [feature for x in ratio], 'borne_inf':res_inf, 'borne_inf_incl':res_inf_incl, 'borne_sup':res_sup, 'borne_sup_incl': res_sup_incl, 'nb_tot': nb, 'ratio': ratio}).drop_duplicates()




def distribImbalanced(df, feature, target, bins=100):
    
    tmp = distribRatio(df, feature, target, bins=bins)
    
    mini = df[feature].min()
    maxi = df[feature].max()
    
    l_inf = [mini + x * (maxi-mini)/bins for x in np.arange(bins)]
    l_sup = [mini + (x+1) * (maxi-mini)/bins for x in np.arange(bins)]
        
    nb_pos_tot = df[df[target] == 1].shape[0]
    nb_tot = df.shape[0]
    ratio_pos = nb_pos_tot / nb_tot

    f = plt.figure() 
    f.set_figwidth(8) 
    f.set_figheight(6) 
    
    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
    plt.plot((tmp.borne_inf + tmp.borne_sup) / 2, tmp.ratio)
    plt.plot([mini,maxi], [ratio_pos for x in [mini,maxi]],'g--')
    plt.xlabel('valeur feature')
    plt.ylabel('ratio classe positive')
    plt.rcParams.update({'font.size':12, 'font.style':'normal'})
    plt.title('Distribution de la classe positive selon ' + feature)
    plt.grid(True)
    plt.show() 






def getStatsAndImp(df):
    
    imp =  evaluateFeatureImportance(df, 
                                    target='TARGET', 
                                    identifier='SK_ID_CURR', 
                                    minmax=[], 
                                    standard=[], 
                                    robust=[], 
                                    ordinal=[], 
                                    onehot=[],
                                    models=[RFmodel, XGmodel],
                                    label_models=['Rando forest','XGBoost'],
                                    show_plot=False,                                     
                                    title_plot='Merged dataset - Feature importance')

    statFeat = stats(df, identifier='SK_ID_CURR',target='TARGET',excludeDominantExtrem=True)
    statFeat = statFeat.merge(imp[['feature','bestClassement','RFClassement','XGClassement']], left_on='feature', right_on='feature')
    statFeat = statFeat.sort_values('bestClassement')
    statFeat = statFeat.reset_index(drop=True)
    
    return statFeat, imp





def Scatter(X, Y, X_name, Y_name, title="", alpha=0.1, color=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig = plt.figure(figsize=(8,8))
    if color is None:
        plt.scatter(X, Y, marker = 'o', alpha = alpha)
    else:
        if type(color[0]) != list:
            plt.scatter(X, Y, marker = 'o', alpha = alpha, c=color)
        else:            
            sc = plt.scatter(X, Y, marker = 'o', alpha = alpha, c=color[0])
            plt.legend(sc.legend_elements()[0], np.unique(color[1]))
    plt.xlabel(X_name)
    plt.ylabel(Y_name)
    plt.rcParams.update({'font.size':15})
    if title == "":
        plt.title(X_name + " vs " + Y_name)
    else:
        plt.title(title)
    plt.show()

    pearson = st.pearsonr(X,Y)[0]
    print("Pearson entre {} et {}: {}".format(X_name, Y_name, np.round(pearson,2)))
    print("R2 entre {} et {}: {}".format(X_name, Y_name, np.round(pearson ** 2,2)))



def metrics_model(trainedModel, x_train, x_test, y_train, y_test, record=False, labelModel="", attributes="", dataSet="", comment=""):

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import r2_score
    ### voir RMSLE
    ### voir RSE, erreur carrée relative qui qui est le complément à 1 du coefficient de détermination R2 (indique une corrélation entre valeurs prédites et vraies valeurs)
    
    exp = False
    
    if exp:
        comment += "EXP"
    
    print("Evaluation sur le jeu d'entrainement")

    start = datetime.datetime.now()
    trainPred = trainedModel.predict(x_train)
    end = datetime.datetime.now()
    duration_train = end - start
    
    if exp:
        pred = np.exp(trainPred)
        real = np.exp(y_train)
    else:
        pred = trainPred
        real = y_train
    

    mae = mean_absolute_error(real, pred)
    print("MAE: {}".format(mae))
        
    R2 = r2_score(real, pred)
    print("R2: {}".format(R2))
    
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    print("RMSE: {}".format(rmse))
    
    pred = [float(np.where(x<0,0,x)) for x in pred]
    rmsle = mean_squared_log_error(real, pred)
    print("RMSLE: {}".format(rmsle))

    
    print("")
    print("Evaluation sur le jeu de test")
    
    start = datetime.datetime.now()
    testPred = trainedModel.predict(x_test)
    end = datetime.datetime.now()
    duration_test = end - start

    if exp:
        pred = np.exp(testPred)
        real = np.exp(y_test)
    else:
        pred = testPred
        real = y_test
        
    mae_test = mean_absolute_error(real, pred)
    print("MAE: {}".format(mae_test))

    R2_test = r2_score(real, pred)
    print("R2: {}".format(R2_test))
    
    mse = mean_squared_error(real, pred)
    rmse_test = np.sqrt(mse)
    print("RMSE: {}".format(rmse_test))
    
    pred = [float(np.where(x<0,0,x)) for x in pred]
    rmsle_test = mean_squared_log_error(real, pred)
    print("RMSLE: {}".format(rmsle_test))

    
    if record:
        iIndex=int(np.where(np.isnan(tableMetrics.index.max()),0, tableMetrics.index.max())) + 1
        tableMetrics.loc[iIndex] = [labelModel, mae, R2, rmse, rmsle, duration_train, mae_test, R2_test, rmse_test, rmsle_test, duration_test, attributes, dataSet, comment]
    


def plot_learning_curves(model, X_train, X_val, y_train, y_val, xscale='', yscale='', legend_loc="upper right"):
    
    from sklearn.metrics import mean_squared_error
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train", color="green" )
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val", color="orange")
    plt.legend(loc=legend_loc, fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown
    if xscale.lower() == "log": plt.xscale('log')
    if yscale.lower() == "log": plt.yscale('log')



def exploreRegulLinReg(X, y, param_alpha=[-5, 5, 200], typeReg="ridge", size=1):
    
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    
    if typeReg.lower() == "ridge":
        model = Ridge()
    elif typeReg.lower() == "lasso":
        model = Lasso(fit_intercept=False)
    
    alphas = np.logspace(param_alpha[0], param_alpha[1], param_alpha[2]) 
    
    coefs = []
    errors = []
    
    for a in alphas:
        model.set_params(alpha=a)
        #ridge.set_params(alpha=a, solver="cholesky")
        # ridge.set_params(alpha=a, solver="sag")
        model.fit(X, y)
        coefs.append(model.coef_)
        errors.append(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
        
    import matplotlib.pyplot as plt

    
    f = plt.figure(figsize=(15,6))
    plt.rcParams.update({'font.size':10})
    
    #plt.title('titre')
    ax = f.add_subplot(121)

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.gca().set_xlabel('alpha')
    plt.gca().set_ylabel('weights')    
    plt.xlabel('alpha')
    plt.ylabel('weights')
    ax.set_title("Coefficients de régression {} en fonction de alpha".format(typeReg))
    plt.axis('tight')
    
    ax2 = f.add_subplot(122)    

    ax2.plot(alphas, errors)
    ax2.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('error')
    plt.axis('tight')
    ax2.set_title("RMSE en fonction de alpha")
    
    plt.rcParams.update({'font.size':15})
    plt.suptitle("Recherche du coefficient de régularisation {} optimal (alpha)".format(typeReg))    
    
    plt.show()



def isnull(value, valueIfNull):
    try:
        test=np.isnan(value)
    except Exception:
        return value
    if np.isnan(value):
        return valueIfNull
    else:
        return value
    

# XXX possible de gérer avec une compréhension de liste?
def rename_col(df):
    """Modifies column names of a dataframe: 'ma colonne'>'MaColonne', '23macolonne'>'c23macolonne' """
    for col in df.columns:
        # if the column starts with a with a number, we add a "c" at the beginning
        if col[:1] in ("0","1","2","3","4","5","6","7","8","9"):
            newCol = 'c' 
        else: 
            newCol = '' 
        # if the column name contains spaces, we remove them and we capitalize the first letter of each word
        if ' ' in col:
            for i,elt in enumerate(col.split(" ")): newCol += elt.capitalize() 
        else:
            newCol += col
        newCol = newCol.replace("-","_").replace("(","_").replace(")","_").replace("/","_")
        df.rename(columns={col:newCol}, inplace=True)
        

        
def showAllHist(df, nbCol=6, size=(20,10)):

    quantCol = myDf(df).Stat(returnTypeCol=True)[1]
    nbQuantCol = len(quantCol)

    nbCol = 6
    nbLignes = int(np.where(nbQuantCol / nbCol == int(nbQuantCol / nbCol),nbQuantCol / nbCol ,int(nbQuantCol / nbCol) + 1 ))

    fig = plt.figure(figsize=size)

    i = 1
    for col in quantCol:
        ax = fig.add_subplot(nbLignes,nbCol,i)
        h = ax.hist(df[col], bins=50, color='steelblue', density=True, edgecolor='none')
        ax.set_title(col, fontsize=10)
        i += 1


        
        
def EraseFile(rep, removeRep=False):

    files=os.listdir(rep)
    for i in range(0,len(files)):
        os.remove(rep+'/'+files[i])

    if removeRep:
        os.removedirs(rep) 
        
        
def displayPictures(url, titre=""):
    
    if not os.path.exists("images"):
        os.makedirs("images")
    EraseFile("images")
    
    urllib.request.urlretrieve(url, "images/tmp.jpg")
    nbPlot = 1
    size = 1
    fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
    
    sub = fig.add_subplot(1,1,1)
    img = mpimg.imread("images/tmp.jpg")
    plt.imshow(img)
    plt.axis('off')
    sub.set_title(titre)
    
    plt.show()

        
def trunc_string(chaine, longueur):
    chaine = str(chaine)
    if len(chaine) < longueur:
        return chaine
    else:
        return chaine[:longueur-1] + ".."
    
def printSingleStat(nb, texte):
    print(str(nb) + " " + texte)


    
def printRatio(nbPart, nbAll, substract=False, text1="", text2=""):
    if substract:
        nb = nbAll-nbPart
    else:
        nb = nbPart
    print(str(nb) + " " + text1 + " " + str(nbAll) + " " + text2 + " (" + str(round(100*(nb)/nbAll,3)) + "%)" )
    
    
def nbCommonWords(string1, string2, lower=True, excludedWords=[]):
    if lower:
        string1 = string1.lower()
        string2 = string2.lower()
    string1 = string1.replace(","," ").replace(";"," ").replace("."," ").replace("-"," ").replace("_"," ").replace("(_)"," ").replace(")"," ")
    string2 = string2.replace(","," ").replace(";"," ").replace("."," ").replace("-"," ").replace("_"," ").replace("(_)"," ").replace(")"," ")
    list1 = string1.split(" ")
    list2 = string2.split(" ")
    for word in excludedWords:
        list1 = removeEltList(list1, word)
        list2 = removeEltList(list2, word)
    return len(commonEltList(list1,list2))


def union(list1,list2):
    liste=[]
    liste.extend(list1)
    liste.extend(list2)
    return liste

    
# retourne les éléments communs à deux listes
def commonEltList(list1, list2):
    return list(set(list1).intersection(list2))

# retourne les éléments d'une liste qui ne sont pas dans une autre
def removeEltList(list1, list2):
    if type(list2) == str:
        list2 = [list2]
    for elt in list2:
        if elt in list1:
            list1.remove(elt)
    return list1
    return list(set(list1)-set(list2)) # on per l'ordre avec ça


def autopctPie(pct,total,thresholdVoid=0, thresholdOnlyPrct=0):
    pct = np.floor(pct * 10) / 10
    if pct < thresholdVoid:
        result = ""
    elif pct < thresholdOnlyPrct:
        result = "{:.1f}%".format(pct)
    else:
        result = "{:.1f}%\n({:d} lignes)".format(pct, int(pct/100.*total))
    return result



def heatMap(table, square=False, annot=False, cmap=sns.diverging_palette(20, 220, n=200), linewidths=.5, colorGradient=[-1,1]):
    
    plt.rcParams.update({'font.size':10})
    if len(colorGradient)==2:
        sns.heatmap(table,square=square,annot=annot, cmap=cmap, linewidths=linewidths, vmin=colorGradient[0], vmax=colorGradient[1])
    else:
        sns.heatmap(table,square=square,annot=annot, cmap=cmap, linewidths=linewidths)
    
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)
    
    # bug fix, see https://github.com/mwaskom/seaborn/issues/1773
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    
    
    

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


def test_3D():

    #%matplotlib notebook

    from mpl_toolkits.mplot3d import Axes3D
    
    matplotlib.interactive(True)

    x, y, z = list(np.random.rand(3,10))

    fig = plt.figure( figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    #ax.set_aspect("equal")


def f_test_quiver():
    x = np.linspace(0,1,11)
    y = np.linspace(0,1,11)
    u = [0,0,0.1,0,0,0.05,0,0,0,0,0]
    v = [0,0,0.2,0,0,0.13,0,0,0,0,0]
    plt.quiver(x, y, u, v, scale=1)        
    
    
# Eboulis
def pcaScreePlot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    #plt.show(block=False)

    
# Cercles de projections
def pcaCircles(pca, axis_ranks=[(0,1),(0,2)], labels=None, label_rotation=0, lims=None, size=1):
    
    #%matplotlib inline
    nbPlot=2
    fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
    numPlot = 1
    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
    
    pcs = pca.components_
    n_comp = len(pcs)

    for d1, d2 in axis_ranks: # affichage des plans factoriels
        if d2 < n_comp:

            # initialisation de la figure
            #fig, ax = plt.subplots(figsize=(7,6))
            sub = fig.add_subplot(1,nbPlot,numPlot)
            numPlot+=1
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Projection 1")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})
            
                
            # détermination des paramètres des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                #fonctionne avec:  fig, ax = plt.subplots(figsize=(7,6))
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # fixation des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.rcParams.update({'font.size':18})
    plt.suptitle("Cercle des projections 2D des variables sur les 3 composantes principales")
        
    plt.show()


# Projection des individus sur les composantes principales
def pca2D(X_projected, pca, axis_ranks=[(0,1),(0,2)], labels=None, multivColorCat=None, alpha=1, size=1):

    #%matplotlib inline
    nbPlot=2
    fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
    numPlot = 1
    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
    
    n_comp = len(pca.components_)
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure
            #fig, ax = plt.subplots(figsize=(7,6))
            sub = fig.add_subplot(1,nbPlot,numPlot)
            numPlot+=1
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Projection 1")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})
        
            # affichage des points
            if multivColorCat is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                labelCat, indexCat = np.unique(multivColorCat, return_inverse=True)
                sc = plt.scatter(X_projected[:, d1], X_projected[:, d2], marker = 'o', c = indexCat, alpha = alpha)
                plt.legend(sc.legend_elements()[0], labelCat)

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            
    plt.rcParams.update({'font.size':18})
    plt.suptitle("Cercle des projections 2D des individus sur les 3 composantes principales")            
    plt.rcParams.update({'font.size':10})
    plt.show()

            

    
def pca3D(X_projected, pca, multivColorCat=None):

    #from importlib import reload
    #reload(matplotlib)
    #from matplotlib import pyplot as plt
    
    #import sys
    #sys.modules.pop('matplotlib')
    #import matplotlib.pyplot as plt
    #%matplotlib notebook
    #matplotlib.interactive(True)
    

    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import FancyArrowPatch

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')    # BUG
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-3,3)
    fig.tight_layout()
    
    x, y, z = X_projected.T
    
    # affichage des points
    if multivColorCat is None:
        ax.scatter(x, y, z, alpha = 0.1, s=1)
    else:
        labelCat, indexCat = np.unique(multivColorCat, return_inverse=True)
        sc = ax.scatter(x, y, z, marker = 'o', c = indexCat, alpha = 0.1)
        plt.legend(sc.legend_elements()[0], labelCat)    
    
    plt.title("PCA 3D")

    # Affichage des vecteurs propres (axes principaux d'inertie)
    pcs = pca.components_
    for i in range(3):
        eigen_vector_x20 = pcs[i] * 20
        eigen_vector = Arrow3D([0, eigen_vector_x20[0]], [0, eigen_vector_x20[1]],[30, 30+eigen_vector_x20[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="#4c72b0", alpha=.7)
        ax.add_artist(eigen_vector)
        ax.text3D(eigen_vector_x20[0],eigen_vector_x20[1],30+eigen_vector_x20[2],"u{} (x20)".format(i+1), color="#333333")
        
    plt.show()
    

            
            

class mySerie(pd.Series):
    
    def Reverse(self):
        
        if self.name == None:
            name = 0
        else:
            name = self.name
            
        return self.to_frame().reset_index().sort_index(ascending=False).set_index('index')[name]


class myDf(pd.DataFrame):

    neutralColor = '#808080'   # '#7f7f7f'
    barChartColor = '#1f77b4'
    barWidth = 0.4
    
    
    def WhereNotExists(self, df, left_on, right_on):
        dfInner = self.merge(df, left_on=left_on, right_on=right_on)[left_on]
        dfInner["testExists"] = 1
        dfLeft = self.merge(df, left_on=left_on, right_on=right_on, how="left")[left_on]
        dfLeft = dfLeft.merge(dfInner, left_on = left_on, right_on = left_on, how="left")
        dfNotExists = dfLeft[dfLeft.testExists.isnull()][left_on].drop_duplicates()
        return self.merge(dfNotExists, left_on=left_on, right_on=right_on)
    

    def GroupBy(self, keys, aggfields="count", resetIndex=True, *restrictions):
            
        restrictions=list(restrictions)
        
        # keys doit être soit une string (qui les clés d'agrégation séparés par une virgule), soit une liste (qui liste les clés)
        if type(keys) == str:
            keys = [key.strip() for key in keys.split(",")]

        # on prépare le nom des colonnes en sortie (par exemple mean_col, max_col...)
        col_result = []        
        if aggfields != "count":
            if type(aggfields) != dict:
                raise TypeError("Le deuxième paramètre doit être 'count' ou un dictionnaire contenant les champs à agréger avec leurs fonctions d'agrégation.")            
            for elt in aggfields:
                if type(aggfields[elt]) != str:
                    raise TypeError("les fonctions d'agrégation doivent être passée dans une chaine de caractères, séparées par des virgules")
                aggfields[elt]  = [aggfield.strip() for aggfield in aggfields[elt].split(",")]
                for fct_aggr in aggfields[elt]:
                    if fct_aggr not in ("count", "min", "max", "sum", "mean", "nunique"):
                        raise ValueError("La fonction d'agrégation " + fct_aggr + " n'est pas applicable")  
                    col_result.append(fct_aggr + "_" + elt)

        # Vérification que ce sont bien des noms de colonne du dataframe qui sont passés en paramètre
        list_to_check = keys.copy()
        if aggfields != "count":
            list_to_check.extend(list(aggfields.keys()))
        for elt in list_to_check:
            if elt not in self.columns:
                raise ValueError(elt + " n'est pas une colonne du dataframe passé en paramètre")  

        result = self.copy()
        
        # clause where
        for where in restrictions:
            try:
                result = self.query(where)
                restrictions.remove(where)
            except Exception:
                pass # si ça plante ici, ça passera peut être plus loin en clause having

        # on ajoute le count de ligne par clé d'agrégation, on boucle sur les colonnes jusqu'à en trouver une qui ne soit pas dans keys (si elle est dans keys ça plante)
        for col in result.columns:
            try:
                result_count = result.groupby(keys).count()[col].to_frame()
                break
            except Exception:
                pass
        result_count.columns = ["count"]
        
        # group by
        if aggfields != "count":
            result = result.groupby(keys).agg(aggfields)
            result.columns = col_result
            
        
        if aggfields != "count":
            result = result.join(result_count)
            col_result.insert(0,"count")
            result = result[col_result]            
        else:
            result = result_count
        
        
        if resetIndex:
            result = result.reset_index(drop = False)
        
        # clause having
        # 
        for having in restrictions:
            try:
                result = result.query(having)
            except Exception:
                raise ValueError(having + " n'est une clause applicable ni en 'where' ni en 'having'")
                
        return result
   

    
    
    def ColCountLines(self, minCount=0, maxCount=9999999999):
        return self.describe(include="all").transpose().query("(count>=" + str(minCount) + ") & (count<=" + str(maxCount) + ")")["count"] / self.shape[0] * 100
    
    
    def ColCountLinesChart(self, typeChart="bar", displayColName=False, minCount=0, maxCount=9999999999):
        plt.rcParams.update({'font.size':10})
        serieGraph = self.ColCountLines(minCount,maxCount)
        # Selon qu'on affiche le nom des colonnes ou pas
        if not displayColName:
            serieGraph = serieGraph.reset_index(drop=True)
        else:
            serieGraph = mySerie(serieGraph)
        # Dimension de l'axe des colonnes (absisses en bar) selon le nombre de colonnes et selon qu'on affiche les noms de colonne ou pas
        if displayColName:
            x_figsize = int(len(serieGraph)/3)
            if typeChart == "bar":
                x_figsize = np.array([np.array([20,x_figsize]).min(), 8]).max()
            else:
                x_figsize = np.array([np.array([13,x_figsize]).min(), 8]).max()
        else:
            x_figsize = 10
        
        if typeChart == "barh":
            fig = plt.figure(figsize=(8, x_figsize))
            plt.barh(y=serieGraph.index,width=serieGraph,height=myDf.barWidth, color=myDf.neutralColor)
            plt.gca().invert_yaxis()
            #plt.gca().set_xlabel('Nombre de lignes')
            if displayColName:
                pass
                #plt.gca().set_ylabel('Nom des colonnes')
            else:
                plt.gca().set_ylabel('Numéro des colonnes')
        else:
            fig = plt.figure(figsize=(x_figsize, 8))
            plt.bar(x=serieGraph.index,height=serieGraph, width=myDf.barWidth, color=myDf.neutralColor)
            #plt.gca().set_ylabel('Nombre de lignes')
            if displayColName:
                pass
                #plt.gca().set_xlabel('Nom des colonnes')
            else:
                plt.gca().set_xlabel('Numéro des colonnes')     
        
        title = "Nombre de lignes renseignées par colonne"
        if (minCount > 0) and (maxCount == 9999999999):
            title += " (au moins " + str(minCount) + " lignes)" 
        elif (minCount == 0) and (maxCount != 9999999999):
            title += " (au plus " + str(maxCount) + " lignes)" 
        elif (minCount > 0) and (maxCount != 9999999999):
            title += " (entre " + str(minCount) + " et " + str(maxCount) + " lignes)" 
        plt.gca().set_title(title)
        
        
        if displayColName:
            plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    
    def WeightMean(self, col, colWeight):
        sumWeight = self[colWeight].sum()
        sumWeightCol = self.apply(lambda x: x[col] * x[colWeight], axis=1).sum()
        return sumWeightCol / sumWeight
    
    
        
    def ClassifyFeature(self,feature, typeFeature="tbd", thresholdContinu=20):
        
        if typeFeature == "tbd":
            dTypeFeature = self.dtypes[feature]
            if dTypeFeature == 'float64': # Si la colonne est float alors que toutes les valeurs sont entières, alors on convertit en entier (pour éviter d'avoir "2.0" dans les libellé et pour identifier une statistique ordinale)        
                typeFeature = 'continue'
                if len(self[feature].unique()) <= thresholdContinu:
                    typeFeature = 'discrete'
                    for index, row in self[[feature]].dropna().drop_duplicates().iterrows():
                        if np.floor(row[feature]) != row[feature]:
                            typeFeature = 'continue'
            elif dTypeFeature== int:
                if len(self[feature].unique()) <= thresholdContinu:
                    typeFeature = 'discrete'
                else:
                    typeFeature = 'continue'
            else:
                # Si les libellés des catégories sont des lettres, on considère qu'on est en ordinal (et par défaut on fait un bar chart)
                if self[[feature]].dropna().apply(lambda x: len(str(x[feature])), axis=1).max() == 1:
                    typeFeature = "ordinale"
                else:
                    typeFeature = "nominale"    

        return pd.DataFrame({"feature": [feature], "typeFeature": [typeFeature]})
                
                

            
        
    def StatUniv(self, col="all", size=1, statsNan=False, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, 
                 lower=False, truncateLabel=35, histDensity=False, histxLog=False, histyLog=False, histBins=100, pieColors="default", 
                 pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1, violinPlot=False, boxShowfliers=True, thresholdContinu=20):

        barh = False
        if typeChart == "barh":
            typeChart = "bar"
            barh = True
           
        # On garde en mémoire si certains champs ont été forcés
        forcedTypeChart = (typeChart != "tbd")
        
        # La série de données à représenter graphiquement
        serieGraph = self[[col]].dropna()
        
        # On passe les noms des modalités en minuscule afin d'éviter les doublons dûs à la casse
        if lower & (serieGraph.dtypes[col] == 'object'):
            serieGraph = serieGraph.apply(lambda x: str(x[col]).strip().lower(), axis=1).to_frame()
            serieGraph.columns = [col]      
            
        # Détermination du type de graph en fonction du type de variable
        if typeChart == "tbd":
            if typeFeature == "nominale":
                typeChart = "pie"
            elif (typeFeature == "ordinale") | (typeFeature == 'discrete'):
                if len(serieGraph[col].unique()) > 3:
                    typeChart = "bar"
                else:
                    typeChart = "pie"
            elif typeFeature == "continue":
                typeChart = "hist"        
        
        
        ############################################################################################
        # Préparation de la série statistique des valeurs non manquantes à représenter graphiquement
        ############################################################################################
        if typeChart == 'hist':
            serieGraphHist = np.array(serieGraph[col])
            
        # Liste des données à mettre en graphique avec pour chaque modalité le nombre de ligne
        serieGraph = serieGraph[col].value_counts()
        nbLigneTotal = sum(serieGraph)
        serieGraph = pd.DataFrame({col:np.array(serieGraph.index), "NbLigne":serieGraph})   
    

                
        ###############################################################
        # Regroupement des petities modalités dans une modalité "autres"
        ###############################################################

        # Dans le cas d'une variable nominale (catégorielle), on gère une modalité "autres" qui regroupe les modalités les moins représentées
        nbBarAutres = 0
        nbLignesAutres = 0
        if (typeFeature == 'nominale') | (typeChart == "pie"):
            # Gestion du threshold qui va servir à regrouper des modalités dans une modalité "autres"
            if type(threshold)==str:
                threshold=threshold.replace(",",".")
                if threshold[-1] != "%":
                    warnings.warn("Si vous passez le threshold en string, écrivez un nombre suivi de '%'. Le threshold est mis à 2,5%.")
                    threshold = 2.5
                else:
                    try:
                        threshold = float(threshold[0:-1])
                    except Exception:
                        warnings.warn(warningThreshold)
                        threshold = 2
            elif (type(threshold)==int) | (type(threshold)==float):
                if (threshold > 0) & (threshold < 1):
                    warnings.warn("Le threshold indiqué semble inférieur à 1%.")
                if (threshold > 100) | (threshold < 0):
                    warnings.warn("Le threshold est un pourcentage et ne doit pas dépasser 100 ou être inférieur à 0. Le threshold est mis à 2,5%.")
                    threshold = 2.5

            # Pour éviter d'avoir trop de modalités représentées dans le graphique, on limite le nombre selon deux critères passés en paramètre:
            #      - threshold (le nombre de ligne que la modalité doit au minimum avoir, en proportion du nombre total de lignes pour lesquelles la modalité est renseignée)
            #      - modalities (nombre de modalités à afficher au maximum)
            # Les lignes qui ne vérifient pas ces conditions sont regroupées dans une ou plusieurs catégories "autre"

            # Si threshold est passé en paramètre et pas modalities, alors on calcule modalities pour qu'il soit en phase avec threshold
            if (threshold != 2.5) & (modalities == 10.1):
                modalities = len(serieGraph.query("NbLigne >= " + str(threshold/100.*nbLigneTotal))) 
            else:
                modalities=int(modalities)
                threshold = 999
            nbLignesAutres = 0  # nombre de lignes des autres modalités
            nbAutresModalites = len(serieGraph) - modalities   # nombre des autres modalités
            # S'il n'y a qu'une autre modalité, on ne la remplace pas une modalité "autres", il faut au moins deux autres modalités pour regrouper
            if nbAutresModalites > 1:
                serieGraphAutres = serieGraph.tail(nbAutresModalites)
                nbLignesAutres = serieGraphAutres.NbLigne.sum()
                maxAutres = serieGraphAutres.head(1).NbLigne.values[0]
                serieGraph = serieGraph.head(modalities)
                # Si la distribution ne contient que des petites modalités, alors la catégorie "autres" prend trop de place et un pie chart n'est plus lisible
                # dans ce cas on passe en bar chart
                if nbLignesAutres / nbLigneTotal > 0.95 and not forcedTypeChart:
                    typeChart = "bar"
                if typeChart=="pie":
                    # Dans le cas d'un pie chart, on regroupe toutes les autres modalités dans une seule portion du pie chart
                    if threshold == 999:
                        # le threshold affiché est le NbLigne (en %) de la modalité la moins courante qui n'est pas dans le groupe "autres"
                        threshold = 100 * maxAutres / nbLigneTotal
                        # on tronque à un chiffre ou plus si il faut) après la virgule
                        if threshold > 0.1:
                            threshold = np.floor(threshold * 10) / 10
                        elif threshold > 0.01:
                            threshold = np.floor(threshold * 100) / 100
                        elif threshold > 0.001:
                            threshold = np.floor(threshold * 1000) / 1000
                        elif threshold > 0.0001:
                            threshold = np.floor(threshold * 10000) / 10000
                        elif threshold > 0.00001:
                            threshold = np.floor(threshold * 100000) / 100000
                        elif threshold > 0.000001:
                            threshold = np.floor(threshold * 1000000) / 1000000
                        else:
                            threshold = 0
                    lblAutre = str(nbAutresModalites) + " autres modalités (<= " + str(threshold) + "%)"
                    serieGraphAutres = pd.DataFrame([[lblAutre,nbLignesAutres]], columns = serieGraph.columns)
                elif typeChart == "bar":
                    # Dans le cas d'un bar chart, on va regrouper les autres modalités dans plusieurs "autres barres"
                    nbBarAutres = 0
                    serieGraphAutres = serieGraphAutres.NbLigne.value_counts().reset_index()
                    serieGraphAutres.columns = ['NbLigne','NbModalities']
                    serieGraphAutres = serieGraphAutres.sort_values("NbLigne", ascending=False)
                    serieGraphAutres = serieGraphAutres.reset_index(drop=True)

                    NbModalities = serieGraphAutres.query("NbLigne == 1").NbModalities.values[0]
                    if NbModalities >= nbAutresModalites / 10:
                        # Si le nombre de modalités à une ligne est significatif, on créé une barre à une ligne
                        nbBarAutres += 1
                        lblAutre = str(NbModalities) + " autres modalités à 1 ligne"
                        serieGraphAutres2 = pd.DataFrame([[lblAutre,1]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.drop(serieGraphAutres[serieGraphAutres.NbLigne==1].index)
                        # Ainsi qu'une barre à deux lignes
                        NbModalities = 0
                        try:
                            NbModalities = serieGraphAutres.query("NbLigne == 2").NbModalities.values[0]
                        except Exception:
                            pass
                        if NbModalities > 0:
                            nbBarAutres += 1
                            lblAutre = str(NbModalities) + " autres modalités à 2 lignes"
                            serieGraphAutres2 = serieGraphAutres2.append(pd.DataFrame([[lblAutre,2]], columns = serieGraph.columns))
                            serieGraphAutres = serieGraphAutres.drop(serieGraphAutres[serieGraphAutres.NbLigne==2].index)
                            # et une barre à trois lignes
                            if len(serieGraphAutres.query("NbLigne == 3")) > 0:
                                NbModalities = serieGraphAutres.query("NbLigne == 3").NbModalities.values[0]
                                if NbModalities > 0:
                                    nbBarAutres += 1
                                    lblAutre = str(NbModalities) + " autres modalités à 3 lignes"
                                    serieGraphAutres2 = serieGraphAutres2.append(pd.DataFrame([[lblAutre,3]], columns = serieGraph.columns))
                                    serieGraphAutres = serieGraphAutres.drop(serieGraphAutres[serieGraphAutres.NbLigne==3].index)

                    if len(serieGraphAutres) > 1000 and (serieGraphAutres.NbLigne.max() / serieGraphAutres.NbLigne.min() > 10):
                        nbBarAutres += 1
                        nbHalfAutre = int(np.floor(len(serieGraphAutres) / 2))
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres.tail(nbHalfAutre)).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.tail(nbHalfAutre).NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres3 = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.head(len(serieGraphAutres) - nbHalfAutre)

                    if len(serieGraphAutres) > 100 and (serieGraphAutres.NbLigne.max() / serieGraphAutres.NbLigne.min() > 10):
                        nbBarAutres += 1
                        nbHalfAutre = int(np.floor(len(serieGraphAutres) / 2))
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres.tail(nbHalfAutre)).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.tail(nbHalfAutre).NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres4 = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.head(len(serieGraphAutres) - nbHalfAutre)

                    if len(serieGraphAutres) > 30 and (serieGraphAutres.NbLigne.max() / serieGraphAutres.NbLigne.min() > 10):
                        nbBarAutres += 1
                        nbHalfAutre = int(np.floor(len(serieGraphAutres) / 2))
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres.tail(nbHalfAutre)).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.tail(nbHalfAutre).NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres5 = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                        serieGraphAutres = serieGraphAutres.head(len(serieGraphAutres) - nbHalfAutre)

                    if len(serieGraphAutres) > 0:
                        nbBarAutres += 1
                        moyLignesAutre = int(np.floor(myDf(serieGraphAutres).WeightMean("NbLigne","NbModalities")))
                        nbAutresModalites = serieGraphAutres.NbModalities.sum()
                        lblAutre = str(nbAutresModalites) + " autres modalités à " + str(moyLignesAutre) + " lignes en moy"
                        serieGraphAutres = pd.DataFrame([[lblAutre,moyLignesAutre]], columns = serieGraph.columns)
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres2)
                    except:
                        pass
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres3)
                    except:
                        pass
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres4)
                    except:
                        pass
                    try:
                        serieGraphAutres = serieGraphAutres.append(serieGraphAutres5)
                    except:
                        pass

                    serieGraphAutres = serieGraphAutres.sort_values("NbLigne", ascending=False)


                serieGraph = serieGraph.append(serieGraphAutres)  

        
        ###################
        # Mise en graphique
        ###################
        # Détermination du nombre de graphiques
        nbPlot = 0
        if statsNan:
            nbPlot += 1 # pour un pie des NaNs
        nbPlot += 1  # pour le graphique principal
        if (typeFeature == "continue") | (typeChart == "hist"):
            nbPlot += 1 # Pour un boxplot
        
        # variable pour les 1 à 4 graphiques
        numPlot = 0
        
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
        
        # S'il n'y a pas de données, on force la représentation des valeurs manquantes et on ne représente pas graphiquement les données
        noData = False
        if len(serieGraph) == 0:
            statsNan=True
            noData = True
            nbPlot = 1
            
        
        # 1 - Affichage d'un pie chart qui représente les valeurs manquantes
        if statsNan:
            numPlot += 1
            sub = fig.add_subplot(1,nbPlot,numPlot)
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Données manquantes vs renseignées")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})
            labels = ["Renseignées (" + str(len(self[col].dropna().unique())) + " modalités)", "Manquantes"]
            renseigne = len(self[self[col].notnull()]) 
            nonRenseigne = len(self[self[col].isna()]) 
            sizes = [renseigne, nonRenseigne]
            #plt.pie(sizes, labels=labels, autopct=lambda pct: autopctPie(pct,renseigne+nonRenseigne), colors = ['#2ca02c', '#d62728'],startangle=90-renseigne/ len(self)*360/2 + 0.0001)
            #plt.pie(sizes, labels=labels, autopct=lambda pct: "{:.1f}%\n({:d} lignes)".format(np.floor(pct * 10) / 10, int(pct/100.*(renseigne+nonRenseigne))), colors = ['#2ca02c', '#d62728'],startangle=90-renseigne/ len(self)*360/2 + 0.0001)            
            plt.pie(sizes, labels=labels, autopct=lambda pct: "{:.1f}%\n({:d} lignes)".format(round(pct * 10) / 10, int(pct/100.*(renseigne+nonRenseigne))), colors = ['#2ca02c', '#d62728'],startangle=90-renseigne/ len(self)*360/2 + 0.0001)            
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            #fig = plt.gcf()
            #fig.gca().add_artist(centre_circle)
            plt.gca().add_artist(centre_circle)
        
        if not noData:

            # 2 - Affichage du graphique principal
            numPlot += 1
            sub = fig.add_subplot(1,nbPlot,numPlot)
            plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            sub.set_title("Distribution des données renseignées")
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})

            if typeChart == "hist":
                serieGraph = serieGraphHist
                plt.hist(serieGraphHist, density=histDensity, bins=histBins)
                plt.xlabel(col)
                if histxLog:
                    plt.xscale('log')
                if histyLog:
                    plt.yscale('log')
                if histDensity == True:
                    plt.ylabel('Fréquence')
                else:
                    plt.ylabel('Nombre de lignes')
                    
            if typeChart=="bar":
                # Si la variable est ordinale, on ordonne selon la modalité (en prenant soin de garder la modalité "autres" à la fin)
                if (typeFeature == "ordinale") | (typeFeature == "discrete"):
                    serieGraph = serieGraph.sort_values(col)
                #serieGraph[col] = serieGraph.apply(lambda x: trunc_string(x[col],truncateLabel), axis=1)
                barColors = [self.barChartColor for _ in np.arange(modalities)]
                if nbBarAutres > 0:
                    barColors.extend([self.neutralColor for _ in np.arange(nbBarAutres)])
                #return serieGraph
                #serieGraph[col] = serieGraph.apply(lambda x: trunc_string(x[col],int(truncateLabel*1.5)), axis=1)
                #plt.bar(x=serieGraph.apply(lambda x: trunc_string(x[col],truncateLabel*1.5), axis=1), height=serieGraph.NbLigne, color=barColors)

                if barh:
                    plt.barh(y=serieGraph.apply(lambda x: str(x[col]), axis=1), width=serieGraph.NbLigne, color=barColors)
                    plt.gca().invert_yaxis()
                else:
                    plt.bar(x=serieGraph.apply(lambda x: str(x[col]), axis=1), height=serieGraph.NbLigne, color=barColors)
                plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
                plt.gca().set_xlabel(col)
                plt.gca().set_ylabel('Nombre de lignes')
                #plt.yscale('log')
                #plt.set_yticklabels(['1e%i' % np.round(np.log(y)/np.log(10)) for y in y_labels])

            if typeChart=="pie":
                
                # On va modifier les données pour l'affichage, on créé une copie (car les données originale vont servir éventuellement plus loin pour le boxplot)
                serieGraphCopy = serieGraph.copy()
                serieGraphCopy[col] = serieGraphCopy.apply(lambda x: trunc_string(x[col],truncateLabel), axis=1)
                serieGraphCopy.at[serieGraphCopy.NbLigne < pieThresholdVoidModality / 100 * nbLigneTotal,[col]] = ""
                labels = serieGraphCopy[col]
                sizes = serieGraphCopy.NbLigne
                # Gestion des couleurs
                if pieColors == "default":  ##0110d0  1f77b4
                    listColors = ['#1f77b4', '#d0c101', '#a83c09']
                    for i in range(10):
                        listColors.extend(['#fc5a50', '#8c564b',   '#9467bd',    '#cf6275', '#e377c2'])
                elif pieColors == "default_plt":# '#380282',
                    listColors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                else:
                    warnings.warn("La couleur passée en paramètre n'est pas 'default' ou 'default_plt', les couleurs seront donc choisies au hasard")
                    listColors=[]
                for i in np.arange(10000):
                    # Au cas où on voudrait mettre mettre 10000 modalités dans un pie, il faut une couleur pour chacune
                    # Et si l'utilisateur saisi autre chose que "default" ou "default_plt", alors toutes les couleurs seront choisies au hasard
                    listColors.append([random.random(),random.random(),random.random()])
                listColors = listColors[0:len(sizes)]
                if nbLignesAutres > 0:
                    listColors = listColors[0:len(sizes)-1]
                    # On affiche "autres" en gris
                    listColors.append(myDf.neutralColor)
                # Setup du pie chart
                #plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=startAngle)
                total = sum(sizes)
                if pieStartAngle == 0.001:
                    nbLigneLastLine = serieGraphCopy.tail(1).NbLigne.values[0]
                    nbLignePrevLastLine = serieGraphCopy.tail(2).head(1).NbLigne.values[0]
                    if nbLigneLastLine > nbLignePrevLastLine:
                        # On est dans le cas où "autres" regroupe beaucoup de lignes ("autre" est dans la dernière ligne de serieGraph)
                        # On a donc beaucoup de modalités avec un petit nombre de lignes
                        # Pour mieux les lire sur le pie chart, on va ramener le plus petite ligne juste au dessus de l'angle 0%
                        # Pour cela, il sufit de remonter la part de camembert "autre" qui se termine à 0%
                        pieStartAngle = 360 * (nbLigneLastLine + nbLignePrevLastLine/2) / nbLigneTotal
                    pieStartAngle = 0 # plus simple comme ça...

                plt.pie(sizes, labels=labels, autopct=lambda pct: autopctPie(pct,total,pieThresholdVoidLabel,pieThresholdOnlyPrctLabel), colors=listColors, startangle=pieStartAngle)

                
            if typeChart == "hist":
  
                mean = np.mean(serieGraph)
                std = np.std(serieGraph)
                mini = np.min(serieGraph)
                maxi = np.max(serieGraph)
                q25 = np.quantile(serieGraph, 0.25)
                q75 = np.quantile(serieGraph, 0.75)
                median = np.median(serieGraph)
                skewness = st.skew(serieGraph)
                kurtosis = st.kurtosis(serieGraph)

                numPlot += 1
                sub = fig.add_subplot(1,nbPlot,numPlot)

                if violinPlot:
                    plt.rcParams.update({'font.size':13, 'font.style':'italic'})
                    sub.set_title("max, min, médiane, quartiles, moyenne")
                    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
                    vp = plt.violinplot(serieGraph, positions=[0],showmedians = False, showmeans = False)
                    color = vp['cbars'].get_color()[0]
                    # plot the IQR, median and mean
                    xx = np.array([-1,1])*0.075 - 0.005
                    plt.fill_between(xx,*np.percentile(serieGraph,[25,75]),color=color,alpha=0.5)
                    #plt.plot(xx,[np.median(serieGraph)]*2,'-',color=color)
                    #plt.plot(xx,[np.mean(serieGraph)]*2,'-',color=self.neutralColor)                    
                    plt.plot(xx,[median]*2,'-',color=color)
                    plt.plot(xx,[mean]*2,'-',color=self.neutralColor)
                    # plot the legend
                    plt.plot([],[],color=self.neutralColor,label='mean')
                    plt.plot([],[],color=color,label='median')
                    plt.fill_between([],[],color=color,label='IQR')
                    if histxLog:
                        plt.yscale('log')
                    if histyLog:
                        plt.xscale('log') 
                    plt.legend()
                else:
                    plt.rcParams.update({'font.size':13, 'font.style':'italic'})
                    sub.set_title("max, min, médiane, quartiles")
                    plt.rcParams.update({'font.size':10, 'font.style':'normal'})
                    sns.boxplot(y=serieGraph,showfliers=boxShowfliers)
                    if histxLog:
                        plt.yscale('log')
                    if histyLog:
                        plt.xscale('log')                    
            
                
            #else:
            #    numPlot += 1
            #    sub = fig.add_subplot(1,nbPlot,numPlot)
            #    serieGraph = np.array(serieGraph_bx.NbLigne)
            #    vp = plt.violinplot(serieGraph, positions=[0],showmedians = False, showmeans = False)   
            #    plt.gca().set_xlabel('Nombre de modalités')
            #    plt.gca().set_ylabel('Nombre de lignes')
            #    plt.rcParams.update({'font.size':13, 'font.style':'italic'})
            #    if serieGraph_bx.NbLigne.min() < 100 * serieGraph_bx.NbLigne.max():
            #        plt.yscale('log')
            #        sub.set_title("Distribution des counts (éch. log.)")
            #    else:
            #        sub.set_title("Distribution des données renseignées")
        
        plt.rcParams.update({'font.size':18})
        if pltTitle == "tbd":
            plt.suptitle("Statistique univariée du champ " + col)
        else:
            plt.suptitle(pltTitle)
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})
            
        plt.show()

        if typeFeature=="continue":
            #plt.axis("off")
            #plt.text(0,1,"Mesures de position")
            print("Mesures de position")
            print(" - Moyenne: " + str(round(mean,5)))
            print(" - Médiane: " + str(round(median,5)))
            print("Mesures de dispertion")
            print(" - Ecart type: " + str(round(std,5)))
            print(" - Min: " + str(round(mini,5)))
            print(" - Q25: " + str(round(q25,5)))
            print(" - Q75: " + str(round(q75,5)))
            print(" - Max: " + str(round(maxi,5)))
            print("Mesures de forme")
            print(" - Skewness (asymétrie): " + str(round(skewness,3)))      
            print(" - Kurtosis (applatissement): " + str(round(kurtosis,3)))
        
        
        
        
    def StatBivQuantQuant(self, features, size=1, statsNan=True, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, histxLog=False, histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1,violinPlot=False, boxShowfliers=True,thresholdContinu=20):

        
        # Détermination du nombre de graphiques
        nbPlot = 3
        
        # variable pour les 1 à 4 graphiques
        numPlot = 0
        
        fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
        
        
        seriesGraph = self[[features[0],features[1]]].dropna()
        
        #print(features)
        #print(seriesGraph)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
            
        # 1 - Affichage d'un scatter plot
        numPlot += 1
        sub = fig.add_subplot(1,nbPlot,numPlot)
        plt.rcParams.update({'font.size':13, 'font.style':'italic'})
        sub.set_title(features[0] + " et " + features[1])
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})
        Y = seriesGraph[features[1]]
        X = seriesGraph[[features[0]]]
        X = X.copy() # On modifiera X, on en crée donc une copie
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
        a,b = result.params[features[0]],result.params['intercept']
        a3 = a
        b3 = b

        plt.scatter(seriesGraph[features[0]],seriesGraph[features[1]], alpha=0.2)
        tmp = np.linspace(seriesGraph[features[0]].min(), seriesGraph[features[0]].max(), 1000)
        plt.plot(tmp,[a*x+b for x in tmp], color='red')
        plt.gca().set_xlabel(features[0])
        plt.gca().set_ylabel(features[1])


        # 2 - Affichage de boites parallèles
        numPlot += 1
        sub = fig.add_subplot(1,nbPlot,numPlot)
        plt.rcParams.update({'font.size':13, 'font.style':'italic'})
        sub.set_title("Distribution de " + features[1] + " par rapport à " + features[0])
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})

        taille_classe = (seriesGraph[features[0]].max() - seriesGraph[features[0]].min()) / 7 # taille des classes pour la discrétisation
        
        # Propriétés graphiques
        medianprops = {'color':"black"}
        meanprops = {'marker':'o', 'markeredgecolor':'black',
                        'markerfacecolor':'firebrick'}

        groupes = [] # va recevoir les données agrégées à afficher

        # on calcule des tranches allant du minimum au maximum par paliers de taille taille_classe
        tranches = np.arange(seriesGraph[features[0]].min(), seriesGraph[features[0]].max(), taille_classe)
        tranches += taille_classe/2 # on décale les tranches d'une demi taille de classe
        #tranches = tranches.round(2)
        indices = np.digitize(seriesGraph[features[0]], tranches) # associe chaque valeur à son numéro de classe

        for ind, tr in enumerate(tranches): # pour chaque tranche, ind reçoit le numéro de tranche et tr la tranche en question
            data = seriesGraph.loc[indices==ind,features[1]] # sélection des lignes de la tranche ind
            if len(data) > 0:
                g = {
                    'valeurs': data,
                    'centre_classe': tr-(taille_classe/2),
                    'taille': len(data),
                    'quartiles': [np.percentile(data,p) for p in [25,50,75]]
                }
                groupes.append(g)

        # affichage des boxplots
        plt.boxplot([g["valeurs"] for g in groupes],
                    positions= [g["centre_classe"] for g in groupes], # abscisses des boxplots
                    showfliers= boxShowfliers, # prise en compte des outliers
                    widths= taille_classe*0.7, # largeur graphique des boxplots
                    showmeans=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True
        )
        plt.xticks(rotation=90)

        for g in groupes:
            plt.text(g["centre_classe"],0,"(n={})".format(g["taille"]),horizontalalignment='center',verticalalignment='top')     
        plt.gca().set_xlabel(features[0])
        plt.gca().set_ylabel(features[1])



        # 3 - Affichage de boites parallèles
        numPlot += 1
        sub = fig.add_subplot(1,nbPlot,numPlot)
        plt.rcParams.update({'font.size':13, 'font.style':'italic'})
        sub.set_title("Distribution de " + features[0] + " par rapport à " + features[1])
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})

        taille_classe = (seriesGraph[features[1]].max() - seriesGraph[features[1]].min()) / 7 # taille des classes pour la discrétisation

        groupes = [] # va recevoir les données agrégées à afficher

        # on calcule des tranches allant de 0 au solde maximum par paliers de taille taille_classe
        tranches = np.arange(seriesGraph[features[1]].min(), seriesGraph[features[1]].max(), taille_classe)
        tranches += taille_classe/2 # on décale les tranches d'une demi taille de classe
        #tranches = tranches.round(2)
        indices = np.digitize(seriesGraph[features[1]], tranches) # associe chaque solde à son numéro de classe

        for ind, tr in enumerate(tranches): # pour chaque tranche, ind reçoit le numéro de tranche et tr la tranche en question
            data = seriesGraph.loc[indices==ind,features[0]] # sélection des individus de la tranche ind
            if len(data) > 0:
                g = {
                    'valeurs': data,
                    'centre_classe': tr-(taille_classe/2),
                    'taille': len(data),
                    'quartiles': [np.percentile(data,p) for p in [25,50,75]]
                }
                groupes.append(g)

        # affichage des boxplots
        plt.boxplot([g["valeurs"] for g in groupes],
                    positions= [g["centre_classe"] for g in groupes], # abscisses des boxplots
                    showfliers= boxShowfliers, # prise en compte des outliers
                    widths= taille_classe*0.7, # largeur graphique des boxplots
                    vert = False,
                    showmeans=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True

        )
        plt.xticks(rotation=90)

        for g in groupes:
            plt.text(0,g["centre_classe"],"(n={})".format(g["taille"]),horizontalalignment='center',verticalalignment='top')     
        plt.gca().set_xlabel(features[0])
        plt.gca().set_ylabel(features[1])


        plt.rcParams.update({'font.size':18})
        if pltTitle == "tbd":
            plt.suptitle("Statistique bivariée des champs " + features[0] + " et " + features[1])
        else:
            plt.suptitle(pltTitle)
        plt.rcParams.update({'font.size':10, 'font.style':'normal'})

        plt.show()


        # Divers prints
        tmp = st.pearsonr(seriesGraph[features[0]],seriesGraph[features[1]])[0] # coeff de correlation linéaire
        print("Covariance entre " + features[0] + " et "  + features[1] + " : {} ".format(np.cov(seriesGraph[features[0]],seriesGraph[features[1]],ddof=0)[1,0]))
        print("Covariance débiaisée entre " + features[0] + " et "  + features[1] + " : {} ".format(np.cov(seriesGraph[features[0]],seriesGraph[features[1]],ddof=1)[1,0]))
        print("Coefficient de correlation linéaire entre " + features[0] + " et "  + features[1] + " : {} ".format(tmp))
        print("Régression linéaire entre " + features[0] + " et "  + features[1] + ": y = {}x + {}".format(a, b))
        print("Coefficient de détermination entre " + features[0] + " et "  + features[1] + " : {} ".format(tmp * tmp))


    def StatBivQualQuant(self, features, size=1, statsNan=True, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, histxLog=False, 
                         histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1,violinPlot=False, boxShowfliers=False, 
                         returnFig=False,
                        thresholdContinu=20):
        
        seriesGraph = self[[features[0],features[1]]].dropna()
        
        # Calcul du coefficient de corrélation
        corr_rapport = eta_squared(seriesGraph[features[0]],seriesGraph[features[1]])
        
        if returnFig:
            return corr_rapport
        else:
            # liste des modalités de la variable qualitative
            modalites = seriesGraph[[features[0]]].sort_values(features[0], ascending=False)[features[0]].unique()

            # On splitte la variable quantitative en fonction de la variable qualitative
            groupes = []
            for m in modalites:
                groupes.append(seriesGraph[seriesGraph[features[0]]==m][features[1]])

            # Propriétés graphiques
            medianprops = {'color':"black"}
            meanprops = {'marker':'o', 'markeredgecolor':'black',
                        'markerfacecolor':'firebrick'}

            # Mise en graphique
            nbPlot = 1
            fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))

            plt.boxplot(groupes, labels=modalites, showfliers=boxShowfliers, medianprops=medianprops, 
                        vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
            plt.gca().set_xlabel(features[1])     
            plt.gca().set_ylabel(features[0])     

            plt.rcParams.update({'font.size':18})
            if pltTitle == "tbd":
                plt.suptitle("Statistique bivariée des champs " + features[0] + " et " + features[1])
            else:
                plt.suptitle(pltTitle)
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})        
            plt.show()      
            
            print("Rapport de corrélation entre {} et {}: {}".format(features[0],features[1],str(corr_rapport)))
    
       
        
    def StatBivQualQual(self, features, size=1, statsNan=True, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, histxLog=False, histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1,violinPlot=False, boxShowfliers=True, returnFig=False,
                        thresholdContinu=20):

        # c : contingence table
        c = self[[features[0],features[1]]].dropna().pivot_table(index=features[0],columns=features[1],aggfunc=len,margins=True,margins_name="Total")
        c = c.fillna(0)

        n = c.loc["Total","Total"]
        tx = c.loc[:,["Total"]]
        ty = c.loc[["Total"],:]
        
        # i : independant table
        i = tx.dot(ty) / n

        chi2Table = (c-i)**2/i
        chi2 = chi2Table.sum().sum()
        
        phi2 = chi2 / n
        r = c.shape[0] - 1
        c = c.shape[1] - 1
        tschuprow = math.sqrt(phi2 / math.sqrt((r-1)*(c-1)))
        cramer = math.sqrt(phi2/(min(r,c)-1))
        
        if returnFig:
            if np.isnan((tschuprow + cramer) / 2):
                return 0
            else:
                return (tschuprow + cramer) / 2
        else:
            nbPlot = 1
            fig = plt.figure(figsize=(size * (18 * nbPlot/2), size * 8))
            table = chi2Table/chi2   
            heatMap(table.iloc[:-1,:-1], colorGradient=[], cmap="YlGnBu")

            plt.rcParams.update({'font.size':18})
            if pltTitle == "tbd":
                plt.suptitle("Statistique bivariée (chi-2) des champs " + features[0] + " et " + features[1])
            else:
                plt.suptitle(pltTitle)
            plt.rcParams.update({'font.size':10, 'font.style':'normal'})        
            plt.show()


            print("chi-2 entre " + features[0] + " et "  + features[1] + " : {} ".format(chi2))
            print("phi-2 entre " + features[0] + " et "  + features[1] + " : {} ".format(phi2))
            print("coefficient T de Tschuprow entre " + features[0] + " et "  + features[1] + " : {} ".format(tschuprow))
            print("coefficient C de Cramer entre " + features[0] + " et "  + features[1] + " : {} ".format(cramer))
 

    def pca(self, nComp=3, reduction=True, labels=None, multivColorCat = None, multivGraph="Circle2D3D"):

    
        # liste des indivdus
        if labels == True:
            labels = self.index
        else:
            labels = None
            
        # Liste des variables
        features = self.columns.tolist()

        # Matrice des valeurs
        X = self.values
        
        # Variable catégorielle pour les couleurs
        if multivColorCat is not None:
            features.remove(multivColorCat)
            multivColor = X[:,-1]
            X = X[:,:-1]
        else:
            multivColor = None

        # Centrage et éventuellement réduction
        X = preprocessing.StandardScaler(with_std=reduction).fit_transform(X)
        # ou en deux temps:
        #X = preprocessing.StandardScaler().fit(X)
        #X = std_scale.transform(X)

        # Calcul des composantes principales
        pca = decomposition.PCA(n_components= nComp)
        pca.fit(X)
        
        # Projection des individus sur les composantes principales
        X_projected = pca.transform(X)
        
        # Eboulis des valeurs propres
        #pcaScreePlot(pca)
        
        # Cercle des corrélations des variables projetées
        if "circle" in multivGraph.lower():
            pcaCircles(pca, [(0,1),(0,2)], labels = np.array(features))
        
        # Visualisation 2D
        if "2d" in multivGraph.lower():
            pca2D(X_projected, pca, alpha = 0.2 ,labels=labels, multivColorCat=multivColor)
        
        # Visualisation 3D
        if "3d" in multivGraph.lower():
            pca3D(X_projected, pca, multivColorCat=multivColor)

 

    def Stat(self, col=["all_columns"], exclCol=[], returnTypeCol=False, bivInclCol=[], bivThresholdCorr=0, bivOnlyCorrTable= False, bivAllCorrTable=True, multivGraph="Circle2D3xD", multivHeatAnnot=True,
             multivDropNa=True, multivMeanNa=False, multivColorCat=None, multivProjNbIndiv=10000,
             typeCol="quantqual", typeStat="univ-biv-multiv", nCompACP=3, size=1, 
             statsNan=False, pltTitle="tbd", typeChart="tbd", typeFeature="tbd", threshold=2.5,  modalities=10.1, lower=False, truncateLabel=35, histDensity=True, 
             histxLog=False, histyLog=False, histBins=100, pieColors="default", pieStartAngle=0.001, pieThresholdVoidLabel=2, pieThresholdOnlyPrctLabel=10, pieThresholdVoidModality=1, 
             violinPlot=False, boxShowfliers=True, returnFig=False, thresholdContinu=20):
        
        ###############################################################
        # listes des statistiques à faire:
        #      - uniariée:un élément string
        #      - bivariée: une liste de deux éléments string
        ###############################################################

        # les champs colonne sont convertis en liste
        if type(col) == str:
            col = col.split(",")
            col = [elt.strip() for elt in col]
        if type(exclCol) == str:
            exclCol = exclCol.split(",")
            exclCol = [elt.strip() for elt in exclCol]
        if type(bivInclCol) == str:
            bivInclCol = bivInclCol.split(",")
            bivInclCol = [elt.strip() for elt in bivInclCol]   
        
        # liste des colonnes sur lesquelles les statistiques vont être faites
        if (col == ["all_columns"]): 
            col = self.columns.to_list()
            col = removeEltList(col, exclCol)   # pour exclure des colonnes
            
        # Les colonnes passées en paramètre doivent être des colonnes du dataframe
        for column in col:
            if column not in self.columns: raise ValueError(column + " n'est pas une colonne du dataframe")
        for column in exclCol:
            if column not in self.columns: raise ValueError(column + " n'est pas une colonne du dataframe")
        for column in bivInclCol:
            if column not in self.columns: raise ValueError(column + " n'est pas une colonne du dataframe")
                    

        # les colonnes de bivInclCol doivent être dans col (pour statistique bivariée)
        for column in bivInclCol:
            if column not in col:
                col.append(column)     
            
        # classification des variables (colonnes)
        features = pd.DataFrame(columns=["feature", "typeFeature"])
        for feature in col:
            features = features.append(self.ClassifyFeature(feature.strip(),typeFeature))            
        features = features.reset_index(drop=True)

        # liste des variables quantitatives
        if typeCol.lower().find("quant") >= 0:
            quantitativeFeatures = features.query("typeFeature == 'continue'").feature.values.tolist()
        else:
            quantitativeFeatures = []
            features = features.drop(features[features.typeFeature == "continue"].index)
            
        # liste des variables qualitatives
        if typeCol.lower().find("qual") >= 0:
            qualitativeFeatures = features.query("typeFeature != 'continue'").feature.values.tolist()
        else:
            qualitativeFeatures = []
            features = features.drop(features[features.typeFeature != "continue"].index)
        
        if returnTypeCol:
            return [qualitativeFeatures,quantitativeFeatures]
        
        # Statistiques univariées
        if typeStat.lower().find("univ") >= 0:
            for index, row in features.iterrows():
                myDf(self).StatUniv(row.feature, statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=row.typeFeature,threshold=threshold,modalities=modalities,lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,histxLog=histxLog,histyLog=histyLog,pieColors=pieColors,pieStartAngle=pieStartAngle,pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)
        
        # Statistiques bivariées
        if typeStat.lower().find("biv") >= 0:        
            
            # bivInclCol ne doit pas mélanger variables qualitatives et quantitatives (sinon les tables de corrélations deviennent difficiles à gérer)
            if (len(commonEltList(bivInclCol,qualitativeFeatures)) > 0) & (len(commonEltList(bivInclCol,quantitativeFeatures)) > 0):
                warnings.warn("Le paramètre bivInclCol ne doit pas mélanger des variables quantitatives et des variables qualitatives.")
                bivInclCol = []                 
            
            corrFeatures = pd.DataFrame(columns=["feature1", "feature2", "typeFeature1", "typeFeature2", "correlation"])
            i=0  # pour incrémenter les index dans corrFeatures
            
            # Corrélations entre variables qualitatives et quantitatives
            for feature1 in qualitativeFeatures:
                for feature2 in quantitativeFeatures:
                    corrRapp = myDf(self).StatBivQualQuant([feature1,feature2],returnFig=True)
                    corrFeatures.loc[i] = [feature1, feature2, "qualitative", "quantitative", corrRapp]
                    i += 1     
            
            # Corrélations entre variables qualitatives
            j=0
            for feature1 in qualitativeFeatures:
                for feature2 in qualitativeFeatures[j:len(qualitativeFeatures)]:
                    if feature1 == feature2:
                        tschuprow = 1
                    else:
                        tschuprow = myDf(self).StatBivQualQual([feature1,feature2],returnFig=True)
                    corrFeatures.loc[i] = [min(feature1, feature2), max(feature1, feature2), "qualitative", "qualitative", tschuprow]
                    i += 1
                j += 1

            # Corrélations entre variables quantitatives
            corrMatrix = self[quantitativeFeatures].corr()
            for index, row in corrMatrix.iterrows():
                for col in corrMatrix.columns:
                    corrFeatures.loc[i] = [min(index, col), max(col, index), "quantitative", "quantitative", row[col]]
                    i += 1
        
            # Suppression des corrélations pour lesquelles aucune variable n'est listée dans bivInclCol
            if bivInclCol != []:
                listIndexToDrop = []
                for index, row in corrFeatures.iterrows():
                    if row.feature1 not in bivInclCol and row.feature2 not in bivInclCol:
                        listIndexToDrop.append(index)
                corrFeatures = corrFeatures.drop(listIndexToDrop)                   
        
            # Suppression des doublons, ordonnancement décroissant en valeur absolue, remplacement des NaN par 1
            corrFeatures = corrFeatures.drop_duplicates()
            corrFeatures["correlationAbs"] = corrFeatures.apply(lambda x: abs(x.correlation), axis = 1)
            corrFeatures = corrFeatures.sort_values("correlationAbs", ascending=False)
            del corrFeatures["correlationAbs"]
            corrFeatures = corrFeatures.fillna(1)
            corrFeatures = corrFeatures.reset_index(drop=True)         
            
            # MATRICE DE TOUTES LES CORRELATIONS
            if bivAllCorrTable:
                if bivOnlyCorrTable | (((len(qualitativeFeatures) * len(quantitativeFeatures)) != 0) & (typeCol != "quant-qual") & (typeCol != "qual-quant")):
                    # Création de la matrice
                    if len(bivInclCol) > 0:
                        inds = bivInclCol
                        cols = bivInclCol + removeEltList(qualitativeFeatures+quantitativeFeatures,bivInclCol)
                    else:
                        inds = qualitativeFeatures+quantitativeFeatures
                        cols = qualitativeFeatures+quantitativeFeatures
                    corrMatrix = pd.DataFrame(index=inds,columns=cols)
                    for index,row in corrFeatures.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation
                    corrMatrix = corrMatrix.fillna(1)

                    # Affichage d'une heat map de la matrice
                    if ((corrMatrix.shape[0] > 2) | (corrMatrix.shape[1] > 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0):
                        f, ax = plt.subplots(figsize=(10, 8))
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})  
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice des corrélations (linéaire,  Tschuprow, rapport de corrélation)")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show()
            
            if not bivOnlyCorrTable:
            
                # STATISTIQUES ENTRE VARIABLES QUALITATIVES

                # Création de la matrice de Tschuprow entre les variables qualitatives
                if len(qualitativeFeatures) >= 2:

                    # Création des colonnes la matrice de Tschuprow
                    if len(bivInclCol) > 0:
                        if len(commonEltList(qualitativeFeatures,bivInclCol)) > 0:
                            inds = commonEltList(qualitativeFeatures,bivInclCol)
                            cols = commonEltList(qualitativeFeatures,bivInclCol) + removeEltList(qualitativeFeatures,bivInclCol)
                        else:
                            inds = []
                            cols = []
                    else:
                        inds = qualitativeFeatures
                        cols = qualitativeFeatures

                    corrTmp = corrFeatures.query("(typeFeature1 == 'qualitative') & (typeFeature2 == 'qualitative')")

                    # Création de la matrice de Tschuprow
                    corrMatrix = pd.DataFrame(index=inds, columns=cols)
                    for index, row in corrTmp.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation 
                    corrMatrix = corrMatrix.fillna(1)               

                    if ((corrMatrix.shape[0] > 2) | (corrMatrix.shape[1] > 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0)  & (typeCol != "quant-qual") & (typeCol != "qual-quant"):
                        # Affichage de la matrice
                        f, ax = plt.subplots(figsize=(10, 8))
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})  
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice de Tschuprow (variables qualitatives)")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show() 

                    # Affichage des statistiques bivariées
                    if  (typeCol != "quant-qual") & (typeCol != "qual-quant"):
                        for index, row in corrTmp.iterrows():
                            if (row["feature1"] != row["feature2"]) & (row["correlation"] >= bivThresholdCorr):
                                myDf(self).StatBivQualQual([row["feature1"], row["feature2"]], statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=typeFeature,threshold=threshold,modalities=modalities,lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,pieColors=pieColors,pieStartAngle=pieStartAngle,pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)


                
                # STATISTIQUES ENTRE VARIABLES QUALITATIVES ET QUANTITATIVES

                # Création de la matrice des rapports de corrélation entre les variables quantitatives et qualitatives
                if (len(quantitativeFeatures) + len(qualitativeFeatures) >= 2) & (len(qualitativeFeatures) > 0) & (len(quantitativeFeatures) > 0):

                    # Création des colonnes la matrice
                    if len(bivInclCol) > 0:
                        inds = bivInclCol
                        if len(commonEltList(quantitativeFeatures,bivInclCol)) > 0:
                            cols = bivInclCol + qualitativeFeatures
                        else:
                            cols = bivInclCol + quantitativeFeatures
                    else:
                        inds = qualitativeFeatures
                        cols = quantitativeFeatures

                    corrTmp = corrFeatures.query("typeFeature1 != typeFeature2")
                    corrTmp = corrTmp.reset_index(drop=True)
                    for index, row in corrFeatures.iterrows():
                        if (row.feature1 in bivInclCol) & (row.feature2 in bivInclCol):
                            corrTmp.loc[len(corrTmp)] = [row.feature1, row.feature2, row.typeFeature1, row.typeFeature1, row.correlation]

                    # Création de la matrice des rapports de corrélation
                    corrMatrix = pd.DataFrame(index=inds, columns=cols)
                    for index, row in corrTmp.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation 

                    corrMatrix = corrMatrix.fillna(1)               

                    if ((corrMatrix.shape[0] >= 2) | (corrMatrix.shape[1] >= 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0):
                        # Affichage de la matrice
                        f, ax = plt.subplots(figsize=(10, 8))
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice des rapports de corrélation entre variables qualitatives et quantitatives")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show() 

                    # Affichage des statistiques bivariées
                    for index, row in corrTmp.iterrows():
                        if (row["feature1"] != row["feature2"]) & (row["typeFeature1"] != row["typeFeature2"]) & (row["correlation"] >= bivThresholdCorr):
                            myDf(self).StatBivQualQuant([row["feature1"], row["feature2"]], statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=typeFeature,threshold=threshold,modalities=modalities,
                                                        lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,pieColors=pieColors,pieStartAngle=pieStartAngle,
                                                        pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,
                                                        violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)

                            
                                            # STATISTIQUES ENTRE VARIABLES QUANTITATIVES

                # Création de la matrice des coefficients de corrélation entre les variables quantitatives
                if len(quantitativeFeatures) >= 2:

                    # Création des colonnes la matrice de Tschuprow
                    if len(bivInclCol) > 0:
                        if len(commonEltList(quantitativeFeatures,bivInclCol)) > 0:
                            inds = commonEltList(quantitativeFeatures,bivInclCol)
                            cols = commonEltList(quantitativeFeatures,bivInclCol) + removeEltList(quantitativeFeatures,bivInclCol)
                        else:
                            inds = []
                            cols = []
                    else:
                        inds = quantitativeFeatures
                        cols = quantitativeFeatures

                    corrTmp = corrFeatures.query("(typeFeature1 == 'quantitative') & (typeFeature2 == 'quantitative')")

                    # Création de la matrice des coefficients de corrélation
                    corrMatrix = pd.DataFrame(index=inds, columns=cols)
                    for index, row in corrTmp.iterrows():
                        if row.feature1 in corrMatrix.index: corrMatrix.at[row.feature1,row.feature2] = row.correlation
                        if row.feature2 in corrMatrix.index: corrMatrix.at[row.feature2,row.feature1] = row.correlation 
                    corrMatrix = corrMatrix.fillna(1)           

                    if ((corrMatrix.shape[0] > 2) | (corrMatrix.shape[1] > 2)) & (corrMatrix.shape[0] * corrMatrix.shape[1] != 0)  & (typeCol != "quant-qual") & (typeCol != "qual-quant"):
                        # Affichage de la matrice
                        f, ax = plt.subplots(figsize=(10, 8))
                        heatMap(corrMatrix, square=True, annot=multivHeatAnnot)
                        plt.rcParams.update({'font.size':18})
                        if pltTitle == "tbd":
                            plt.suptitle("Matrice des coefficients de corrélation (variables quantitatives)")
                        else:
                            plt.suptitle(pltTitle)
                        plt.rcParams.update({'font.size':10, 'font.style':'normal'})                
                        plt.show()

                    # Affichage des statistiques bivariées
                    if  (typeCol != "quant-qual") & (typeCol != "qual-quant"): 
                        for index, row in corrTmp.iterrows():             
                            if (row["feature1"] != row["feature2"]) & (abs(row["correlation"]) >= bivThresholdCorr):
                                myDf(self).StatBivQuantQuant([row["feature1"], row["feature2"]], statsNan=statsNan,pltTitle=pltTitle,typeChart=typeChart,typeFeature=typeFeature,threshold=threshold,modalities=modalities,lower=lower,truncateLabel=truncateLabel,histDensity=histDensity,histBins=histBins,pieColors=pieColors,pieStartAngle=pieStartAngle,pieThresholdVoidLabel=pieThresholdVoidLabel,pieThresholdOnlyPrctLabel=pieThresholdOnlyPrctLabel,pieThresholdVoidModality=pieThresholdVoidModality,violinPlot=violinPlot, boxShowfliers=boxShowfliers,thresholdContinu=thresholdContinu)

                                
                                
        # Statistique mutlivariée (PCA sur 3 composantes principales)
        if typeStat.lower().find("multiv") >= 0: 

            if len(quantitativeFeatures) >= 3:
                
                data_pca = self.copy()
                columnsPca = quantitativeFeatures
                
                if multivColorCat is not None:
                    if multivColorCat in self.columns:
                        data_pca = data_pca.dropna(subset=[multivColorCat])
                        columnsPca.append(multivColorCat)
                    else:
                        warnings.warn(multivColorCat + " n'est pas une colonne du dataframe")
                        multivColorCat = None

                data_pca = data_pca[columnsPca]
                                                       
                if multivDropNa:
                    data_pca = data_pca.dropna(subset=quantitativeFeatures)
                elif multivMeanNa:
                    data_pca[quantitativeFeatures] = self[quantitativeFeatures].fillna(data_pca.mean()) 
                    
                # On prend max 10000 lignes
                data_pca = data_pca.reset_index(drop=True)
                multivProjNbIndiv = min(multivProjNbIndiv,data_pca.shape[0])
                data_pca = data_pca.iloc[random.sample(data_pca.index.tolist(), min(multivProjNbIndiv,len(data_pca)))]

                myDf(data_pca).pca(multivColorCat=multivColorCat, multivGraph=multivGraph)
    
            
            
                        
                


      
