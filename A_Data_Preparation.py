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




t1 = time.time()

print(datetime.datetime.now())


#########################################
# Data for training or dashboard
#########################################
train_or_test = "test"

#########################################
# CHARGEMENT DES FICHIERS
#########################################

dfApplication = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/application_' + train_or_test + '.csv',",")
dfBureau = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/bureau.csv',",")
dfBureauBalance = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/bureau_balance.csv',",")
dfPreviousApplication = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/previous_application.csv',",")
dfPosCashBalance = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/POS_CASH_balance.csv',",")
dfCreditCardBalance = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/credit_card_balance.csv',",")
dfInstallmentsPayments = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/installments_payments.csv',",")

t2 = time.time()
print("{} - {} - fichiers chargés".format(datetime.datetime.now(), t2-t1))
t1 = t2


#########################################
# MARQUAGE DES VALEURS MANQUANTES
#########################################

dfApplication = dfApplication.replace('XNA', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('XNA', np.nan)
dfBureau = dfBureau.replace('XNA', np.nan)
dfApplication = dfApplication.replace('XAP', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('XAP', np.nan)
dfBureau = dfBureau.replace('XAP', np.nan)
dfBureauBalance = dfBureauBalance.replace('X', np.nan)

dfApplication = dfApplication.replace('Unknown', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('Unknown', np.nan)
dfBureau = dfBureau.replace('Unknown', np.nan)
dfApplication = dfApplication.replace('Unknown', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('Unknown', np.nan)
dfBureau = dfBureau.replace('Unknown', np.nan)



t2 = time.time()
print("{} - {} - marquage des valeurs manquantes".format(datetime.datetime.now(), t2-t1))
t1 = t2




#########################################
# TRANSFORMATION DE VARIABLES
#########################################

# On remonte le dernier statut des tables de niveau 3 dans PreviousApplication

dfPreviousApplicationStatus = pd.concat([dfPosCashBalance[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','NAME_CONTRACT_STATUS']],dfCreditCardBalance[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','NAME_CONTRACT_STATUS']]])
tmp = dfPreviousApplicationStatus.groupby(['SK_ID_CURR','SK_ID_PREV']).agg({'MONTHS_BALANCE':'max'}).reset_index(drop=False)
dfPreviousApplicationStatus = dfPreviousApplicationStatus.merge(tmp, left_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], right_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
del dfPreviousApplicationStatus['MONTHS_BALANCE']
dfPreviousApplicationStatus = dfPreviousApplicationStatus.rename(columns={'NAME_CONTRACT_STATUS': 'LAST_NAME_CONTRACT_STATUS'})
dfPreviousApplication = dfPreviousApplication.merge(dfPreviousApplicationStatus, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on=['SK_ID_CURR','SK_ID_PREV'], how='left')

dfPreviousApplication.at[dfPreviousApplication[dfPreviousApplication.LAST_NAME_CONTRACT_STATUS == 'Completed'].index, "LAST_NAME_CONTRACT_STATUS_TMP"] = 0
dfPreviousApplication.at[dfPreviousApplication[(dfPreviousApplication.LAST_NAME_CONTRACT_STATUS != 'Completed') & (~dfPreviousApplication.LAST_NAME_CONTRACT_STATUS.isnull())].index, "LAST_NAME_CONTRACT_STATUS_TMP"] = 1
dfPreviousApplication['LAST_NAME_CONTRACT_STATUS'] = dfPreviousApplication['LAST_NAME_CONTRACT_STATUS_TMP']
del dfPreviousApplication['LAST_NAME_CONTRACT_STATUS_TMP']
dfPreviousApplication['LAST_NAME_CONTRACT_STATUS'] = dfPreviousApplication['LAST_NAME_CONTRACT_STATUS'].astype(float)

del dfCreditCardBalance['NAME_CONTRACT_STATUS']
del dfPosCashBalance['NAME_CONTRACT_STATUS']


t2 = time.time()
print("{} - {} - remontée de name_contract_status".format(datetime.datetime.now(), t2-t1))
t1 = t2




#########################################
# RENOMMAGE DE COLONNES
#########################################

def newColNames(df, suffix):
    cols = []
    keys = ['SK_ID_CURR','SK_ID_PREV','SK_ID_BUREAU']
    for col in df:
        if col not in keys:
            col = suffix + '_' + col
        cols.append(col)
    return cols
newColNames(dfPreviousApplication, 'PREV')


dfPreviousApplication.columns = newColNames(dfPreviousApplication, 'PREV')
dfPosCashBalance.columns = newColNames(dfPosCashBalance, 'POSCASH')
dfInstallmentsPayments.columns = newColNames(dfInstallmentsPayments, 'INSTALPAYMT')
dfCreditCardBalance.columns = newColNames(dfCreditCardBalance, 'CREDCARD')
dfBureau.columns = newColNames(dfBureau, 'BURO')
dfBureauBalance.columns = newColNames(dfBureauBalance, 'BUROBAL')



t2 = time.time()
print("{} - {} - suppression de variables et renommage de colonnes".format(datetime.datetime.now(), t2-t1))
t1 = t2



#########################################
# MISE A PLAT DES DONNEES
#########################################


dfBureauBalance = TransformUnique(dfBureauBalance, ['SK_ID_BUREAU'])
dfPosCashBalance = TransformUnique(dfPosCashBalance, ['SK_ID_CURR','SK_ID_PREV'])
dfCreditCardBalance = TransformUnique(dfCreditCardBalance, ['SK_ID_CURR','SK_ID_PREV'])
dfInstallmentsPayments = TransformUnique(dfInstallmentsPayments, ['SK_ID_CURR','SK_ID_PREV'])

dfPreviousApplication = dfPreviousApplication.merge(dfCreditCardBalance, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on = ['SK_ID_CURR','SK_ID_PREV'], how='left')
dfPreviousApplication = dfPreviousApplication.merge(dfPosCashBalance, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on = ['SK_ID_CURR','SK_ID_PREV'], how='left')
dfPreviousApplication = dfPreviousApplication.merge(dfInstallmentsPayments, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on = ['SK_ID_CURR','SK_ID_PREV'], how='left')

dfBureau = dfBureau.merge(dfBureauBalance, left_on='SK_ID_BUREAU', right_on = 'SK_ID_BUREAU', how='left')

dfPreviousApplication = TransformUnique(dfPreviousApplication, ['SK_ID_CURR'])
dfBureau = TransformUnique(dfBureau, ['SK_ID_CURR'])


dfApplication = dfApplication.merge(dfPreviousApplication, left_on='SK_ID_CURR', right_on = 'SK_ID_CURR', how='left')
dfApplication = dfApplication.merge(dfBureau, left_on='SK_ID_CURR', right_on = 'SK_ID_CURR', how='left')


t2 = time.time()
print("{} - {} - mise à plat".format(datetime.datetime.now(), t2-t1))
t1 = t2



#########################################
# Suppression des colonnes ID
#########################################


del dfApplication['SK_ID_PREV']
del dfApplication['SK_ID_BUREAU']
del dfApplication['INSTALPAYMT_NUM_INSTALMENT_VERSION']
del dfApplication['INSTALPAYMT_NUM_INSTALMENT_NUMBER']


#########################################
# SUPPRESSION DES COLONNES SANS INFORMATION
#########################################

counts = dfApplication.nunique()
counts = counts[counts==1]
to_del = list(counts.index)
dfApplication.drop(to_del, axis=1, inplace=True)


t2 = time.time()
print("{} - {} - suppression de colonnes".format(datetime.datetime.now(), t2-t1))
t1 = t2

#########################################
# GESTION DES QUELQUES VALEURS NUMERIQUES MANQUANTES
#########################################

dfApplication['OWN_CAR_AGE'] = dfApplication['OWN_CAR_AGE'].replace(np.nan, 100)
dfApplication['PREV_LAST_NAME_CONTRACT_STATUS'] = dfApplication['PREV_LAST_NAME_CONTRACT_STATUS'].replace(np.nan, -1)

for col in dfApplication.columns:
    if (col[0:4] in ['PREV','POSC','CRED','INST','BURO']) & (dfApplication[col].dtypes in ['int64','float64']):
        dfApplication[col] = dfApplication[col].replace(np.nan, 0)


dfApplicationDefault = dfApplication

t2 = time.time()
print("{} - {} - missing default".format(datetime.datetime.now(), t2-t1))
t1 = t2

if False:
    del dfApplicationDefault['EXT_SOURCE_1']
    del dfApplicationDefault['EXT_SOURCE_2']
    del dfApplicationDefault['EXT_SOURCE_3']








t1 = time.time()

print(datetime.datetime.now())

#########################################
# CHARGEMENT DES FICHIERS
#########################################

dfApplication = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/application_' + train_or_test + '.csv',",")
dfBureau = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/bureau.csv',",")
dfBureauBalance = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/bureau_balance.csv',",")
dfPreviousApplication = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/previous_application.csv',",")
dfPosCashBalance = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/POS_CASH_balance.csv',",")
dfCreditCardBalance = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/credit_card_balance.csv',",")
dfInstallmentsPayments = pd.read_csv(os.getcwd() + '/Projet+Mise+en+prod+-+home-credit-default-risk/installments_payments.csv',",")

t2 = time.time()
print("{} - {} - fichiers chargés".format(datetime.datetime.now(), t2-t1))
t1 = t2


#########################################
# MARQUAGE DES VALEURS MANQUANTES
#########################################

dfApplication = dfApplication.replace('XNA', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('XNA', np.nan)
dfBureau = dfBureau.replace('XNA', np.nan)
dfApplication = dfApplication.replace('XAP', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('XAP', np.nan)
dfBureau = dfBureau.replace('XAP', np.nan)
dfBureauBalance = dfBureauBalance.replace('X', np.nan)

dfApplication = dfApplication.replace('Unknown', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('Unknown', np.nan)
dfBureau = dfBureau.replace('Unknown', np.nan)
dfApplication = dfApplication.replace('Unknown', np.nan)
dfPreviousApplication = dfPreviousApplication.replace('Unknown', np.nan)
dfBureau = dfBureau.replace('Unknown', np.nan)


t2 = time.time()
print("{} - {} - marquage des valeurs manquantes".format(datetime.datetime.now(), t2-t1))
t1 = t2


#########################################
# SUPPRESSION DE LIGNES
#########################################

dfPreviousApplication = dfPreviousApplication[(dfPreviousApplication.FLAG_LAST_APPL_PER_CONTRACT == 'Y') & (dfPreviousApplication.NFLAG_LAST_APPL_IN_DAY == 1)]
dfPreviousApplication = dfPreviousApplication[dfPreviousApplication.NAME_CONTRACT_STATUS.isin(['Approved','Refused'])]
dfInstallmentsPayments = dfInstallmentsPayments[dfInstallmentsPayments.AMT_INSTALMENT != 0]


t2 = time.time()
print("{} - {} - suppression de lignes".format(datetime.datetime.now(), t2-t1))
t1 = t2



#########################################
# SUPPRESSION DE VARIABLES
#########################################


if False:

    # APPLICATION

    del dfApplication['OBS_60_CNT_SOCIAL_CIRCLE'] # corr 1 avec OBS_60_CNT_SOCIAL_CIRCLE
    del dfApplication['DEF_60_CNT_SOCIAL_CIRCLE'] # corr 0.86 avec DEF_30_CNT_SOCIAL_CIRCLE
    del dfApplication['OWN_CAR_AGE'] # corr 0.97 avec FLAG_OWN_CAR
    del dfApplication['CNT_FAM_MEMBERS'] # corr 0.97 avec CNT_CHILDREN
    del dfApplication['AMT_GOODS_PRICE'] # corr 0.99 avec AMT_CREDIT
    del dfApplication['REGION_RATING_CLIENT_W_CITY'] # corr 0.96 avec REGION_RATING_CLIENT


    # Trop de valeurs identiques
    del dfApplication['FLAG_MOBIL']
    del dfApplication['FLAG_CONT_MOBILE']
    del dfApplication['REG_REGION_NOT_LIVE_REGION']
    for col in dfApplication.columns:
        if (col[0:13] == 'FLAG_DOCUMENT') & (col[-2:] not in ['_3','_6','_8']):
            del dfApplication[col]


    # PREVIOUS APPLICATION

    # Trop de valeurs manquantes
    del dfPreviousApplication['RATE_INTEREST_PRIMARY']
    del dfPreviousApplication['RATE_INTEREST_PRIVILEGED']
    del dfPreviousApplication['NAME_CASH_LOAN_PURPOSE']

    # Trop de valeurs identiques
    del dfPreviousApplication['NAME_PAYMENT_TYPE']

    # corr > 0.9 avec AMT_CREDIT
    del dfPreviousApplication['AMT_GOODS_PRICE']

    # corr 0.9 avec NAME_CONTRACT_TYPE
    del dfPreviousApplication['NAME_PORTFOLIO']

    # corr > 0.95 avec DAYS_DECISION
    del dfPreviousApplication['DAYS_FIRST_DRAWING']
    del dfPreviousApplication['DAYS_FIRST_DUE']
    del dfPreviousApplication['DAYS_LAST_DUE_1ST_VERSION']
    del dfPreviousApplication['DAYS_LAST_DUE']
    del dfPreviousApplication['DAYS_TERMINATION']


    # Champs purement techniques
    del dfInstallmentsPayments['NUM_INSTALMENT_VERSION']
    del dfInstallmentsPayments['NUM_INSTALMENT_NUMBER']


    # Trop de valeurs identiques
    del dfBureau['CREDIT_CURRENCY']
    del dfBureau['CNT_CREDIT_PROLONG']
    del dfBureau['CREDIT_DAY_OVERDUE']
    del dfBureau['AMT_CREDIT_SUM_OVERDUE']

    # Fortement corrélés à DAYS_CREDIT
    del dfBureau['DAYS_CREDIT_UPDATE']
    del dfBureau['DAYS_ENDDATE_FACT']


    # Mauvaise qualité de données
    del dfBureau['DAYS_CREDIT_ENDDATE']


    # Trop souvent manquant et ne peut pas se déduire d'autres champs
    del dfBureau['AMT_ANNUITY']


    
#########################################
# TRANSFORMATION DE VARIABLES
#########################################

# APPLICATION


# Recalcul des champs REQ
dfApplication['AMT_REQ_CREDIT_BUREAU_YEAR'] = dfApplication.AMT_REQ_CREDIT_BUREAU_HOUR + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_DAY + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_WEEK + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_MON + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_QRT + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_YEAR

dfApplication['AMT_REQ_CREDIT_BUREAU_QRT'] = dfApplication.AMT_REQ_CREDIT_BUREAU_HOUR + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_DAY + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_WEEK + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_MON + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_QRT

dfApplication['AMT_REQ_CREDIT_BUREAU_MON'] = dfApplication.AMT_REQ_CREDIT_BUREAU_HOUR + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_DAY + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_WEEK + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_MON

dfApplication['AMT_REQ_CREDIT_BUREAU_WEEK'] = dfApplication.AMT_REQ_CREDIT_BUREAU_HOUR + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_DAY + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_WEEK

dfApplication['AMT_REQ_CREDIT_BUREAU_DAY'] = dfApplication.AMT_REQ_CREDIT_BUREAU_HOUR + \
                                                dfApplication.AMT_REQ_CREDIT_BUREAU_DAY



t2 = time.time()
print("{} - {} - feature engineering sur REQ".format(datetime.datetime.now(), t2-t1))
t1 = t2






# Sans emploi = O DAYS_EMPLOYED
days_employed = dfApplication['DAYS_EMPLOYED'].values
days_employed = [np.min([0,x]) for x in days_employed]
dfApplication['DAYS_EMPLOYED'] = pd.Series(days_employed)



t2 = time.time()
print("{} - {} - feature engineering sur days_employed".format(datetime.datetime.now(), t2-t1))
t1 = t2


# Création d'une variable indiquant si la personne vit seule
dfApplication.at[dfApplication[dfApplication.NAME_FAMILY_STATUS == 'Married'].index, "SINGLE_FLAG"] = 0
dfApplication.at[dfApplication[dfApplication.NAME_FAMILY_STATUS == 'Civil marriage'].index, "SINGLE_FLAG"] = 0
dfApplication.at[dfApplication[dfApplication.NAME_FAMILY_STATUS == 'Single / not married'].index, "SINGLE_FLAG"] = 1
dfApplication.at[dfApplication[dfApplication.NAME_FAMILY_STATUS == 'Separated'].index, "SINGLE_FLAG"] = 1
dfApplication.at[dfApplication[dfApplication.NAME_FAMILY_STATUS == 'Widow'].index, "SINGLE_FLAG"] = 1
dfApplication.at[dfApplication[dfApplication.NAME_FAMILY_STATUS == 'Unknown'].index, "SINGLE_FLAG"] = 1
dfApplication['SINGLE_FLAG'] = dfApplication['SINGLE_FLAG'].astype(float)


t2 = time.time()
print("{} - {} - feature engineering sur FAMILY_STATUS".format(datetime.datetime.now(), t2-t1))
t1 = t2



# Variables living: on compte combien de nuls
livingCols = ['SK_ID_CURR']
for col in dfApplication.columns:
    if col[-4:] in ['MEDI','_AVG','MODE']:
        livingCols.append(col)
dfLiving = dfApplication[livingCols].copy()
dfLiving['NB_LIVING_NOT_PROVIDED'] = dfLiving.apply(lambda x: x.isnull().sum(), axis='columns').astype('int64')
dfLiving = dfLiving[['SK_ID_CURR','NB_LIVING_NOT_PROVIDED']]
dfApplication = dfApplication.merge(dfLiving, left_on='SK_ID_CURR', right_on='SK_ID_CURR')
if False:
    for col in dfApplication.columns:
        if col[-4:] in ['MEDI','_AVG','MODE']:
            del dfApplication[col]

del dfLiving 


t2 = time.time()
print("{} - {} - feature engineering sur champs MODE/MEDI/AVG".format(datetime.datetime.now(), t2-t1))
t1 = t2



# Variables documents: on compte combien de documents fournis
docCols = ['SK_ID_CURR']
for col in dfApplication.columns:
    if col.find("DOCUM") > 0:
        docCols.append(col)
dfDoc = dfApplication[docCols].copy()
dfDoc['NB_DOC_FURNISHED'] = dfDoc.sum(axis=1) - dfDoc['SK_ID_CURR']
dfDoc['NB_DOC_FURNISHED'] = dfDoc['NB_DOC_FURNISHED'].astype('int64')
dfDoc = dfDoc[['SK_ID_CURR','NB_DOC_FURNISHED']]
dfApplication = dfApplication.merge(dfDoc, left_on='SK_ID_CURR', right_on='SK_ID_CURR')
if False:
    for col in dfApplication.columns:
        if col.find("DOCUM") > 0:
            del dfApplication[col]     

del dfDoc


t2 = time.time()
print("{} - {} - feature engineering sur champs documents".format(datetime.datetime.now(), t2-t1))
t1 = t2




# PREV APPLICATION ET BUREAU



# Creation d'une variable donnant les écart de temps entre les crédits précédents
#https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering
df1 = dfBureau[['SK_ID_CURR','SK_ID_BUREAU','DAYS_CREDIT']]
df1.columns = ['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION']
df2 = dfPreviousApplication[['SK_ID_CURR','SK_ID_PREV','DAYS_DECISION']]
dfDays = pd.concat([df1,df2])
grp = dfDays[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_DECISION']].groupby(by = ['SK_ID_CURR'])
grp1 = grp.apply(lambda x: x.sort_values(['DAYS_DECISION'], ascending = False)).reset_index(drop = True)#rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})
grp1['DAYS_DECISION1'] = grp1['DAYS_DECISION']*-1
grp1['PREVBURO_DAYS_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_DECISION1'].diff()
grp1['PREVBURO_DAYS_DIFF'] = grp1['PREVBURO_DAYS_DIFF'].fillna(0).astype('uint32')
del grp1['DAYS_DECISION1'], grp1['DAYS_DECISION'], grp1['SK_ID_CURR']
dfDays = dfDays.merge(grp1, left_on='SK_ID_PREV', right_on='SK_ID_PREV')
del dfDays['DAYS_DECISION']
del dfDays['SK_ID_PREV']
dfDays = dfDays.groupby('SK_ID_CURR').mean().reset_index(drop=False)
dfDays['PREVBURO_DAYS_DIFF'] = dfDays['PREVBURO_DAYS_DIFF'].astype(float)

dfApplication = dfApplication.merge(dfDays, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

del dfDays


t2 = time.time()
print("{} - {} - PREV DAYS_DIFF".format(datetime.datetime.now(), t2-t1))
t1 = t2







# PREV APPLICATION

# Création d'une variable CONTRACT_STATUS

# On remonte le dernier statut des tables de niveau 3 dans PreviousApplication
dfPreviousApplicationStatus = pd.concat([dfPosCashBalance[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','NAME_CONTRACT_STATUS']],dfCreditCardBalance[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','NAME_CONTRACT_STATUS']]])
tmp = dfPreviousApplicationStatus.groupby(['SK_ID_CURR','SK_ID_PREV']).agg({'MONTHS_BALANCE':'max'}).reset_index(drop=False)
dfPreviousApplicationStatus = dfPreviousApplicationStatus.merge(tmp, left_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], right_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
del dfPreviousApplicationStatus['MONTHS_BALANCE']
dfPreviousApplicationStatus = dfPreviousApplicationStatus.rename(columns={'NAME_CONTRACT_STATUS': 'LAST_NAME_CONTRACT_STATUS'})
dfPreviousApplication = dfPreviousApplication.merge(dfPreviousApplicationStatus, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on=['SK_ID_CURR','SK_ID_PREV'], how='left')

dfPreviousApplication['CONTRACT_STATUS'] = dfPreviousApplication.apply(lambda x: np.where(x.NAME_CONTRACT_STATUS=='Refused','Refused',np.where(x.LAST_NAME_CONTRACT_STATUS=='Completed','Completed','Active')), axis=1)
del dfPreviousApplication['NAME_CONTRACT_STATUS']
del dfPreviousApplication['LAST_NAME_CONTRACT_STATUS']

del dfPreviousApplicationStatus

t2 = time.time()
print("{} - {} - gestion du contract_status".format(datetime.datetime.now(), t2-t1))
t1 = t2



# Depuis quand date le dernier crédit encore actif?
dfCreditLastDay = pd.concat([dfBureau[dfBureau.CREDIT_ACTIVE == 'Active'][['SK_ID_CURR','DAYS_CREDIT']],dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS == 'Active'][['SK_ID_CURR','DAYS_DECISION']].rename(columns={'DAYS_DECISION':'DAYS_CREDIT'})])
dfCreditLastDay.columns = ['SK_ID_CURR','PREVBURO_LAST_DAYS_DECISION']
dfCreditLastDay = dfCreditLastDay.groupby('SK_ID_CURR').max().reset_index(drop=False)
dfApplication = dfApplication.merge(dfCreditLastDay, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['PREVBURO_LAST_DAYS_DECISION'] = dfApplication['PREVBURO_LAST_DAYS_DECISION'].fillna(0).astype(float)
del dfCreditLastDay

t2 = time.time()
print("{} - {} - last days decision".format(datetime.datetime.now(), t2-t1))
t1 = t2



# Remontée dans dfApplication du nombre de previous active, completed et refused
dfPreviousRefused = dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS == 'Refused']
dfNbPreviousRefused = pd.DataFrame(dfPreviousRefused.groupby('SK_ID_CURR').size(), columns=['PREV_NB_REFUSED']).reset_index(drop=False)
dfPreviousActive = dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS == 'Active']
dfNbPreviousActive = pd.DataFrame(dfPreviousActive.groupby('SK_ID_CURR').size(), columns=['PREV_NB_ACTIVE']).reset_index(drop=False)
dfPreviousCompleted = dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS == 'Completed']
dfNbPreviousCompleted = pd.DataFrame(dfPreviousCompleted.groupby('SK_ID_CURR').size(), columns=['PREV_NB_COMPLETED']).reset_index(drop=False)
dfApplication = dfApplication.merge(dfNbPreviousRefused, right_on='SK_ID_CURR', left_on='SK_ID_CURR', how='left')
dfApplication = dfApplication.merge(dfNbPreviousActive, right_on='SK_ID_CURR', left_on='SK_ID_CURR', how='left')
dfApplication = dfApplication.merge(dfNbPreviousCompleted, right_on='SK_ID_CURR', left_on='SK_ID_CURR', how='left')

del dfNbPreviousRefused
del dfNbPreviousActive
del dfNbPreviousCompleted


t2 = time.time()
print("{} - {} - remonée nb active, completed, refused".format(datetime.datetime.now(), t2-t1))
t1 = t2


# Séparation des refused et des completed/active
dfPreviousApplicationRefused = dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS == 'Refused']
dfPreviousApplication = dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS != 'Refused']
del dfPreviousApplication['CODE_REJECT_REASON']

# Remontée dans dfApplication du code de rejet le plus courant pour chaque client
dfPreviousApplicationRefused = ColMode(dfPreviousApplicationRefused, ['CODE_REJECT_REASON'], ['SK_ID_CURR'])
dfPreviousApplicationRefused = dfPreviousApplicationRefused.rename(columns={'CODE_REJECT_REASON':'PREV_CODE_REJECT_REASON'})
dfApplication = dfApplication.merge(dfPreviousApplicationRefused, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['PREV_CODE_REJECT_REASON'] = dfApplication['PREV_CODE_REJECT_REASON'].fillna('none')
del dfPreviousApplicationRefused

t2 = time.time()
print("{} - {} - remontée de code_reject_reason".format(datetime.datetime.now(), t2-t1))
t1 = t2


# Remontée dans dfApplication des montants previous
# Crédits remboursés
dfPrevAmtCompleted = dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS == 'Completed'][['SK_ID_CURR','AMT_CREDIT']].groupby('SK_ID_CURR').agg({'AMT_CREDIT':'sum'}).reset_index(drop=False).rename(columns={'AMT_CREDIT':'PREV_AMT_COMPLETED'})
dfApplication = dfApplication.merge(dfPrevAmtCompleted, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['PREV_AMT_COMPLETED'] = dfApplication['PREV_AMT_COMPLETED'].fillna(0).astype('float')
del dfPrevAmtCompleted
# Crédits revolving actifs
dfPrevAmtRevolving = dfPreviousApplication[(dfPreviousApplication.CONTRACT_STATUS == 'Active') & (dfPreviousApplication.NAME_CONTRACT_TYPE == 'Revolving loans')][['SK_ID_CURR','AMT_CREDIT']].groupby('SK_ID_CURR').agg({'AMT_CREDIT':'sum'}).reset_index(drop=False).rename(columns={'AMT_CREDIT':'PREV_AMT_REVOLVING'})
dfApplication = dfApplication.merge(dfPrevAmtRevolving, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['PREV_AMT_REVOLVING'] = dfApplication['PREV_AMT_REVOLVING'].fillna(0).astype('float')
del dfPrevAmtRevolving
# Crédits classiques actifs: 1 - remontée du montant total dans application
dfPrevAmtCreditActive = dfPreviousApplication[(dfPreviousApplication.CONTRACT_STATUS == 'Active') & (dfPreviousApplication.NAME_CONTRACT_TYPE != 'Revolving loans')][['SK_ID_CURR','AMT_CREDIT']].groupby('SK_ID_CURR').agg({'AMT_CREDIT':'sum'}).reset_index(drop=False).rename(columns={'AMT_CREDIT':'PREV_AMT_CREDIT_ACTIVE'})
dfApplication = dfApplication.merge(dfPrevAmtCreditActive, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['PREV_AMT_CREDIT_ACTIVE'] = dfApplication['PREV_AMT_CREDIT_ACTIVE'].fillna(0).astype('float')
# Crédits classiques actifs: 2 - remontée dans previous du nombre de paiement effectué
dfPaymentsDone = pd.DataFrame(dfInstallmentsPayments[['SK_ID_PREV','SK_ID_CURR']].groupby(['SK_ID_PREV','SK_ID_CURR']).size(), columns=['CNT_PAYMENT_DONE']).reset_index(drop=False)
dfPreviousApplication = dfPreviousApplication.merge(dfPaymentsDone, left_on=['SK_ID_PREV','SK_ID_CURR'], right_on=['SK_ID_PREV','SK_ID_CURR'], how='left')
dfPreviousApplication['CNT_PAYMENT_DONE'] = dfPreviousApplication['CNT_PAYMENT_DONE'].fillna(0).astype('float')
del dfPaymentsDone
# Crédits classiques actifs: 3 - estimation d'un ratio d'avancement du remboursement
dfPreviousApplication['RATIO_CNT_PAYMENT'] = dfPreviousApplication.apply(lambda x: np.where(x.CONTRACT_STATUS=='Completed', 1, np.where(x.CNT_PAYMENT==0, 0, x.CNT_PAYMENT_DONE/(x.CNT_PAYMENT+0.1))), axis=1)
# Crédits classiques actifs: 4 - application du ratio à AMT_CREDIT
dfPreviousApplication['AMT_CREDIT_RATIO'] = dfPreviousApplication.apply(lambda x: x.RATIO_CNT_PAYMENT * x.AMT_CREDIT, axis=1)
# Crédits classiques actifs: 5 - calcul d'un indicateur global par client du niveau de remboursement de ses crédits actifs
tmp = dfPreviousApplication[(dfPreviousApplication.CONTRACT_STATUS == 'Active') & (dfPreviousApplication.NAME_CONTRACT_TYPE != 'Revolving loans')]
tmp = tmp[['SK_ID_CURR','AMT_CREDIT_RATIO']].groupby('SK_ID_CURR').agg({'AMT_CREDIT_RATIO':'sum'}).reset_index(drop=False).rename(columns={'AMT_CREDIT_RATIO':'SUM_AMT_CREDIT_RATIO'})
dfPrevAmtCreditActive = dfPrevAmtCreditActive.merge(tmp, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfPrevAmtCreditActive['PREV_AMT_CREDIT_ACTIVE_REIMBURSED_INDIC'] = dfPrevAmtCreditActive.apply(lambda x: x.SUM_AMT_CREDIT_RATIO / x.PREV_AMT_CREDIT_ACTIVE , axis=1)
dfPrevAmtCreditActive = dfPrevAmtCreditActive[['SK_ID_CURR','PREV_AMT_CREDIT_ACTIVE_REIMBURSED_INDIC']]
dfApplication = dfApplication.merge(dfPrevAmtCreditActive, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
del dfPrevAmtCreditActive
dfApplication['PREV_AMT_CREDIT_ACTIVE_REIMBURSED_INDIC'] = dfApplication['PREV_AMT_CREDIT_ACTIVE_REIMBURSED_INDIC'].fillna(0).astype('float')
#Annuité des crédits actifs
dfPrevAmtAnnuity = dfPreviousApplication[dfPreviousApplication.CONTRACT_STATUS == 'Active'][['SK_ID_CURR','AMT_ANNUITY']]
dfPrevAmtAnnuity = dfPrevAmtAnnuity.groupby('SK_ID_CURR').agg({'AMT_ANNUITY':'sum'}).reset_index(drop=False).rename(columns={'AMT_ANNUITY':'PREV_AMT_ANNUITY_ACTIVE'})
dfApplication = dfApplication.merge(dfPrevAmtAnnuity, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['PREV_AMT_ANNUITY_ACTIVE'] = dfApplication['PREV_AMT_ANNUITY_ACTIVE'].fillna(0)
dfApplication['PREV_AMT_ANNUITY_ACTIVE'] = dfApplication['PREV_AMT_ANNUITY_ACTIVE'].astype('float')
del dfPrevAmtAnnuity

cols = ['CNT_PAYMENT_DONE', 'RATIO_CNT_PAYMENT', 'AMT_CREDIT_RATIO']
for col in cols:
    del dfPreviousApplication[col]
    
t2 = time.time()
print("{} - {} - remonté dans application des montants de previous".format(datetime.datetime.now(), t2-t1))
t1 = t2



# Création d'une variable ratio Annuity / Income
dfApplication = dfApplication.reset_index(drop=True)
annuity = dfApplication.AMT_ANNUITY.values
prev_annuity = dfApplication.PREV_AMT_ANNUITY_ACTIVE.values
income = dfApplication.AMT_INCOME_TOTAL.values
ratio_annuity_income = (prev_annuity + annuity) / income
dfApplication['RATIO_ANNUITY_INCOME'] = pd.Series(ratio_annuity_income).astype(float)

t2 = time.time()
print("{} - {} - calcul ratio annuity/income".format(datetime.datetime.now(), t2-t1))
t1 = t2


# On ordonne NAME_YIELD_GROUP
dfPreviousApplication.at[dfPreviousApplication[dfPreviousApplication.NAME_YIELD_GROUP == 'high'].index, "NAME_YIELD_GROUP_ORD"] = 3
dfPreviousApplication.at[dfPreviousApplication[dfPreviousApplication.NAME_YIELD_GROUP == 'middle'].index, "NAME_YIELD_GROUP_ORD"] = 2
dfPreviousApplication.at[dfPreviousApplication[dfPreviousApplication.NAME_YIELD_GROUP == 'low_normal'].index, "NAME_YIELD_GROUP_ORD"] = 1
dfPreviousApplication.at[dfPreviousApplication[dfPreviousApplication.NAME_YIELD_GROUP == 'low_action'].index, "NAME_YIELD_GROUP_ORD"] = 1
dfPreviousApplication['NAME_YIELD_GROUP_ORD'] = dfPreviousApplication['NAME_YIELD_GROUP_ORD'].astype(float)  

t2 = time.time()
print("{} - {} - PREV NAME_YIELD_GROUP".format(datetime.datetime.now(), t2-t1))
t1 = t2



# Numérisation de NAME_PRODUCT_TYPE
dfXSell = dfPreviousApplication[dfPreviousApplication.NAME_PRODUCT_TYPE == 'x-sell'][['SK_ID_CURR']].drop_duplicates()
dfXSell['X_SELL'] = 1
dfPreviousApplication = dfPreviousApplication.merge(dfXSell, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfPreviousApplication['X_SELL'] = dfPreviousApplication['X_SELL'].fillna(0).astype(float)

t2 = time.time()
print("{} - {} - PREV XSELL".format(datetime.datetime.now(), t2-t1))
t1 = t2





# BUREAU

# Remontée dans dfApplication du montant total de crédits actifs
tmp = dfBureau[dfBureau.CREDIT_ACTIVE == 'Active'][['SK_ID_CURR','AMT_CREDIT_SUM']].groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM':'sum'}).reset_index(drop=False).rename(columns={'AMT_CREDIT_SUM':'BURO_AMT_CREDIT_ACTIVE'})
dfApplication = dfApplication.merge(tmp, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['BURO_AMT_CREDIT_ACTIVE'] = dfApplication['BURO_AMT_CREDIT_ACTIVE'].fillna(0).astype(float)

# Remontée dans dfApplication du montant total de crédits cloturés
tmp = dfBureau[dfBureau.CREDIT_ACTIVE == 'Closed'][['SK_ID_CURR','AMT_CREDIT_SUM']].groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM':'sum'}).reset_index(drop=False).rename(columns={'AMT_CREDIT_SUM':'BURO_AMT_CREDIT_COMPLETED'})
dfApplication = dfApplication.merge(tmp, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
dfApplication['BURO_AMT_CREDIT_COMPLETED'] = dfApplication['BURO_AMT_CREDIT_COMPLETED'].fillna(0).astype(float)

del tmp


t2 = time.time()
print("{} - {} - remonté montants BURO".format(datetime.datetime.now(), t2-t1))
t1 = t2



# INSTALLMENTS

# Création de variables d'écart entre les montants et échéances attendus et réels
diff_days = dfInstallmentsPayments.DAYS_INSTALMENT.values - dfInstallmentsPayments.DAYS_ENTRY_PAYMENT.values
diff_amt = (dfInstallmentsPayments.AMT_INSTALMENT.values - dfInstallmentsPayments.AMT_PAYMENT.values) / dfInstallmentsPayments.AMT_INSTALMENT.values
days_entry = dfInstallmentsPayments.DAYS_ENTRY_PAYMENT.values
diff_days_pos = [np.max([x,0]) for x in diff_days] / np.sqrt(1-days_entry)
diff_days_neg = [np.min([x,0]) for x in diff_days] / np.sqrt(1-days_entry)
diff_amt_pos = [np.max([x,0]) for x in diff_amt] / np.sqrt(1-days_entry)
diff_amt_neg = [np.min([x,0]) for x in diff_amt] / np.sqrt(1-days_entry)

dfInstallmentsPayments = dfInstallmentsPayments.reset_index(drop=True)
dfInstallmentsPayments['INSTALPAYMT_DIFF_DAYS_INSTALLMENT_PAYMENT_POS'] = pd.Series(diff_days_pos)
dfInstallmentsPayments['INSTALPAYMT_DIFF_DAYS_INSTALLMENT_PAYMENT_NEG'] = pd.Series(diff_days_neg)
dfInstallmentsPayments['INSTALPAYMT_DIFF_AMT_INSTALLMENT_PAYMENT_POS'] = pd.Series(diff_amt_pos)
dfInstallmentsPayments['INSTALPAYMT_DIFF_AMT_INSTALLMENT_PAYMENT_NEG'] = pd.Series(diff_amt_neg)

dfInstallmentsPayments = dfInstallmentsPayments[['SK_ID_CURR','INSTALPAYMT_DIFF_DAYS_INSTALLMENT_PAYMENT_POS','INSTALPAYMT_DIFF_DAYS_INSTALLMENT_PAYMENT_NEG','INSTALPAYMT_DIFF_AMT_INSTALLMENT_PAYMENT_POS','INSTALPAYMT_DIFF_AMT_INSTALLMENT_PAYMENT_NEG']]
dfInstallmentsPayments = dfInstallmentsPayments.groupby('SK_ID_CURR').mean().reset_index(drop=False)
dfApplication = dfApplication.merge(dfInstallmentsPayments, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

del dfInstallmentsPayments
del diff_days
del diff_amt
del days_entry
del diff_days_pos
del diff_days_neg
del diff_amt_pos
del diff_amt_neg



t2 = time.time()
print("{} - {} - INSTALLMENT".format(datetime.datetime.now(), t2-t1))
t1 = t2



# POS CASH

# Remontée des DPD dans dfApplication
# On donne moins d'importance aux Days Past Due anciens


pos_cash_dpd = dfPosCashBalance['SK_DPD'].values
pos_cash_dpd_def = dfPosCashBalance['SK_DPD_DEF'].values
pos_cash_month_bal = dfPosCashBalance['MONTHS_BALANCE'].values
pos_cash_dpd = pos_cash_dpd / np.sqrt(1-pos_cash_month_bal) 
pos_cash_dpd_def = pos_cash_dpd_def / np.sqrt(1-pos_cash_month_bal) 

dfPosCashBalance = dfPosCashBalance.reset_index(drop=True)
dfPosCashBalance['POSCASH_SK_DPD'] = pd.Series(pos_cash_dpd)
dfPosCashBalance['POSCASH_SK_DPD_DEF'] = pd.Series(pos_cash_dpd_def)

dfPosCashDPD = dfPosCashBalance[['SK_ID_CURR','POSCASH_SK_DPD','POSCASH_SK_DPD_DEF']]
dfPosCashDPD = dfPosCashDPD.groupby('SK_ID_CURR').mean().reset_index(drop=False)

dfApplication = dfApplication.merge(dfPosCashDPD, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

dfApplication['POSCASH_SK_DPD'] = dfApplication['POSCASH_SK_DPD'].fillna(0)
dfApplication['POSCASH_SK_DPD_DEF'] = dfApplication['POSCASH_SK_DPD_DEF'].fillna(0)


del dfPosCashDPD
del pos_cash_dpd
del pos_cash_dpd_def
del pos_cash_month_bal


t2 = time.time()
print("{} - {} - POSCASH DPD".format(datetime.datetime.now(), t2-t1))
t1 = t2


tmp = dfPosCashBalance[dfPosCashBalance.NAME_CONTRACT_STATUS.isin(['Active', 'Completed'])]
tmp = dfPosCashBalance.groupby(['SK_ID_CURR','SK_ID_PREV']).agg({'MONTHS_BALANCE':'max'}).reset_index(drop=False)
tmp = dfPosCashBalance.merge(tmp, left_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], right_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
tmp = tmp[['SK_ID_CURR','CNT_INSTALMENT_FUTURE']].groupby('SK_ID_CURR').agg({'CNT_INSTALMENT_FUTURE':'sum'}).reset_index(drop=False)
tmp = tmp.rename(columns={'CNT_INSTALMENT_FUTURE':'POSCASH_CNT_INSTALMENT_FUTURE_LEFT'})
dfApplication = dfApplication.merge(tmp, left_on='SK_ID_CURR', right_on='SK_ID_CURR')

tmp = dfPosCashBalance[dfPosCashBalance.NAME_CONTRACT_STATUS.isin(['Active', 'Completed'])]
tmp = dfPosCashBalance.groupby(['SK_ID_CURR','SK_ID_PREV']).agg({'MONTHS_BALANCE':'min'}).reset_index(drop=False)
tmp = dfPosCashBalance.merge(tmp, left_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'], right_on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
tmp = tmp[['SK_ID_CURR','CNT_INSTALMENT_FUTURE']].groupby('SK_ID_CURR').agg({'CNT_INSTALMENT_FUTURE':'sum'}).reset_index(drop=False)
tmp = tmp.rename(columns={'CNT_INSTALMENT_FUTURE':'POSCASH_CNT_INSTALMENT_FUTURE_TOTAL'})
dfApplication = dfApplication.merge(tmp, left_on='SK_ID_CURR', right_on='SK_ID_CURR')

dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_DONE'] = dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_TOTAL'] - dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_LEFT']
dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_DONE'] = dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_DONE'].fillna(0)
dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_LEFT'] = dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_LEFT'].fillna(0)


del dfApplication['POSCASH_CNT_INSTALMENT_FUTURE_TOTAL']
del dfPosCashBalance['CNT_INSTALMENT_FUTURE']


del dfPosCashBalance




t2 = time.time()
print("{} - {} - POSCASH CNT_INSTALMENT_FUTURE".format(datetime.datetime.now(), t2-t1))
t1 = t2



# CREDIT CARD BALANCE

# Remontée des DPD dans dfApplication
# On donne moins d'importance aux Days Past Due anciens


cred_card_dpd = dfCreditCardBalance['SK_DPD'].values
cred_card_dpd_def = dfCreditCardBalance['SK_DPD_DEF'].values
cred_card_month_bal = dfCreditCardBalance['MONTHS_BALANCE'].values
cred_card_dpd = cred_card_dpd / np.sqrt(1-cred_card_month_bal) 
cred_card_dpd_def = cred_card_dpd_def / np.sqrt(1-cred_card_month_bal) 

dfCreditCardBalance = dfCreditCardBalance.reset_index(drop=True)
dfCreditCardBalance['CREDCARD_SK_DPD'] = pd.Series(cred_card_dpd)
dfCreditCardBalance['CREDCARD_SK_DPD_DEF'] = pd.Series(cred_card_dpd_def)

dfCreditCardDPD = dfCreditCardBalance[['SK_ID_CURR','CREDCARD_SK_DPD','CREDCARD_SK_DPD_DEF']]
dfCreditCardDPD = dfCreditCardDPD.groupby('SK_ID_CURR').mean().reset_index(drop=False)

dfApplication = dfApplication.merge(dfCreditCardDPD, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

dfApplication['CREDCARD_SK_DPD'] = dfApplication['CREDCARD_SK_DPD'].fillna(0)
dfApplication['CREDCARD_SK_DPD_DEF'] = dfApplication['CREDCARD_SK_DPD_DEF'].fillna(0)

del dfCreditCardBalance
del dfCreditCardDPD
del cred_card_dpd
del cred_card_dpd_def
del cred_card_month_bal






t2 = time.time()
print("{} - {} - CREDITCARD".format(datetime.datetime.now(), t2-t1))
t1 = t2



# BUREAU BALANCE


t2 = time.time()
print("{} - {} - mise à zéro des status balance bureau C et X".format(datetime.datetime.now(), t2-t1))
t1 = t2

dfBureauBalance = dfBureauBalance.merge(dfBureau[['SK_ID_BUREAU','SK_ID_CURR']], left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU').reset_index(drop=True)


# Création d'une variable indiquant le niveau de DPD (day past due) du client sur Bureau
# On donne moins d'importance aux Days Past Due anciens

dfBureauBalance.at[dfBureauBalance[dfBureauBalance.STATUS.isin(['C','X'])].index, 'STATUS'] = '0'
dfBureauBalance['STATUS'] = dfBureauBalance['STATUS'].astype(float)

buro_bal_status = dfBureauBalance['STATUS'].values
buro_bal_month_bal = dfBureauBalance['MONTHS_BALANCE'].values
buro_bal_status = buro_bal_status / np.sqrt(1-buro_bal_month_bal) 

dfBureauBalance = dfBureauBalance.reset_index(drop=True)
dfBureauBalance['STATUS'] = pd.Series(buro_bal_status)

dfBureauBalance = dfBureauBalance.rename(columns={'STATUS':'BUROBAL_SK_DPD'})
del dfBureauBalance['SK_ID_BUREAU']
del dfBureauBalance['MONTHS_BALANCE']


dfBureauBalance = dfBureauBalance.groupby('SK_ID_CURR').sum().reset_index(drop=False)

dfApplication = dfApplication.merge(dfBureauBalance, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

dfApplication['BUROBAL_SK_DPD'] = dfApplication['BUROBAL_SK_DPD'].fillna(0)

del dfBureauBalance
del buro_bal_status
del buro_bal_month_bal



t2 = time.time()
print("{} - {} - BUROBAL".format(datetime.datetime.now(), t2-t1))
t1 = t2






#########################################
# RENOMMAGE DE COLONNES
#########################################

def newColNames(df, suffix):
    cols = []
    keys = ['SK_ID_CURR','SK_ID_PREV','SK_ID_BUREAU']
    for col in df:
        if col not in keys:
            col = suffix + '_' + col
        cols.append(col)
    return cols

dfPreviousApplication.columns = newColNames(dfPreviousApplication, 'PREV')
#dfPosCashBalance.columns = newColNames(dfPosCashBalance, 'POSCASH')
#dfInstallmentsPayments.columns = newColNames(dfInstallmentsPayments, 'INSTALPAYMT')
#dfCreditCardBalance.columns = newColNames(dfCreditCardBalance, 'CREDCARD')
dfBureau.columns = newColNames(dfBureau, 'BURO')
#dfBureauBalance.columns = newColNames(dfBureauBalance, 'BUROBAL')



t2 = time.time()
print("{} - {} - renommage de colonnes".format(datetime.datetime.now(), t2-t1))
t1 = t2



#########################################
# MISE A PLAT DES DONNEES
#########################################



#dfBureauBalance = TransformUnique(dfBureauBalance, ['SK_ID_BUREAU'])
#dfPosCashBalance = TransformUnique(dfPosCashBalance, ['SK_ID_CURR','SK_ID_PREV'])
#dfCreditCardBalance = TransformUnique(dfCreditCardBalance, ['SK_ID_CURR','SK_ID_PREV'])
#dfInstallmentsPayments = TransformUnique(dfInstallmentsPayments, ['SK_ID_CURR','SK_ID_PREV'])

#dfPreviousApplication = dfPreviousApplication.merge(dfCreditCardBalance, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on = ['SK_ID_CURR','SK_ID_PREV'], how='left')
#dfPreviousApplication = dfPreviousApplication.merge(dfPosCashBalance, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on = ['SK_ID_CURR','SK_ID_PREV'], how='left')
#dfPreviousApplication = dfPreviousApplication.merge(dfInstallmentsPayments, left_on=['SK_ID_CURR','SK_ID_PREV'], right_on = ['SK_ID_CURR','SK_ID_PREV'], how='left')
#dfBureau = dfBureau.merge(dfBureauBalance, left_on='SK_ID_BUREAU', right_on = 'SK_ID_BUREAU', how='left')

dfPreviousApplication = TransformUnique(dfPreviousApplication, ['SK_ID_CURR'])
dfBureau = TransformUnique(dfBureau, ['SK_ID_CURR'])

dfApplication = dfApplication.merge(dfPreviousApplication, left_on='SK_ID_CURR', right_on = 'SK_ID_CURR', how='left')
dfApplication = dfApplication.merge(dfBureau, left_on='SK_ID_CURR', right_on = 'SK_ID_CURR', how='left')


t2 = time.time()
print("{} - {} - mise à plat".format(datetime.datetime.now(), t2-t1))
t1 = t2



#########################################
# Suppression des colonnes ID
#########################################

#del dfApplication['SK_ID_PREV']


#########################################
# SUPPRESSION DES COLONNES SANS INFORMATION
#########################################

counts = dfApplication.nunique()
counts = counts[counts==1]
to_del = list(counts.index)
dfApplication.drop(to_del, axis=1, inplace=True)


t2 = time.time()
print("{} - {} - suppression de colonnes sans information".format(datetime.datetime.now(), t2-t1))
t1 = t2


#########################################
# GESTION DES QUELQUES VALEURS NUMERIQUES MANQUANTES
#########################################

cols = ['PREV_NFLAG_INSURED_ON_APPROVAL',
        'PREV_X_SELL']
dfApplication[cols] = dfApplication[cols].replace(np.nan, -1)


for col in dfApplication.columns:
    if (col[0:4] in ['PREV','POSC','CRED','INST','BURO']) & (dfApplication[col].dtypes in ['int64','float64']):
        dfApplication[col] = dfApplication[col].replace(np.nan, 0)
        
# Trop de valeurs manquantes
del dfApplication['PREV_RATE_INTEREST_PRIMARY']
del dfApplication['PREV_RATE_INTEREST_PRIVILEGED']
del dfApplication['PREV_NAME_CASH_LOAN_PURPOSE']
        
        
t2 = time.time()
print("{} - {} - missing values".format(datetime.datetime.now(), t2-t1))
t1 = t2

if False:
    del dfApplication['EXT_SOURCE_1']
    del dfApplication['EXT_SOURCE_2']
    del dfApplication['EXT_SOURCE_3']

del dfApplication ['SK_ID_PREV']
del dfApplication ['SK_ID_BUREAU']



for col in dfApplicationDefault.columns:
    if col not in dfApplication.columns:
        dfApplication = dfApplication.merge(dfApplicationDefault[['SK_ID_CURR',col]])

if train_or_test == 'train':
    dump(dfApplication, open('dfApplicationTrain.pkl','wb'))
else:
    imp = load(open('imp.pkl','rb'))
    pipeline = load(open('pipeline.pkl','rb'))
    dfApplication['SCORE'] = pipeline.predict_proba(dfApplication[imp]).T[1]
    cols = ['SK_ID_CURR','SCORE']
    cols.extend(imp)
    dfApplication = dfApplication[cols]
    dump(dfApplication, open('dfApplicationDash.pkl','wb'))

