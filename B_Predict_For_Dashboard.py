from pickle import load, dump
import joblib
from A_Data_Preparation import prepareData

# Préparation des données
dfApplication = prepareData('test')
#dfApplication = load(open('dfApplicationTest.pkl','rb'))

# Load du modèle
#pipeline = load(open('pipeline.pkl','rb'))        
pipeline = joblib.load('pipeline.joblib')
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
