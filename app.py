from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
from sklearn.pipeline import Pipeline
from data_preparation import prepareData
from pickle import load, dump

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    authorized = False
    if request.method == 'POST':
        if request.form['code'] == '1234':
            run_model(request.form['data_size'])
            authorized = True
    return render_template('index.html', authorized=authorized)


@app.route('/a', methods=['GET','POST'])
def test():
    run_model()
    return 'Done!'



@app.route('/<int:sk_id_curr>', methods=['GET'])
def get_client_data(sk_id_curr):
    dfApplicationDash_col = load(open('pickle/dfApplicationDash_col.pkl','rb'))
    dfApplicationDash_value = load(open('pickle/dfApplicationDash_value.pkl','rb'))
    threshold = load(open('pickle/threshold.pkl','rb'))
    df = pd.DataFrame(dfApplicationDash_value, columns = dfApplicationDash_col)
    df['DEFAULT_RISK'] = df.apply(lambda x: 1 if x.SCORE > threshold else 0, axis=1)
    df = df[['SK_ID_CURR','SCORE','DEFAULT_RISK']]
    df = df[df.SK_ID_CURR == sk_id_curr]
    df = df.reset_index(drop=True)
    if df.shape[0] == 0:
        dico = {}
    else:
        dico = df.to_dict()
        for key, value in dico.items():
            dico[key] = value[0]
    return jsonify(dico)



def run_model(data_size):

    # Préparation des données
    dfApplication = prepareData('test', data_size)

    # Load du modèle
    pipeline = load(open('pickle/pipeline.pkl','rb'))

    # Prédiction du score (Les colonnes du dataframe doivent être dans l'ordre attendu par le modèle)
    imp = load(open('pickle/feature_importance.pkl','rb'))   
    dfApplication['SCORE'] = pipeline.predict_proba(dfApplication[list(imp.feature)]).T[1]

    # On ordonne les colonnes selon leur importance
    cols = ['SK_ID_CURR','SCORE']
    cols.extend(list(imp.feature))
    dfApplication = dfApplication[cols]

    # Sauvegarde
    dump(list(dfApplication.columns), open('pickle/dfApplicationDash_col.pkl','wb'))
    dump(list(dfApplication.values), open('pickle/dfApplicationDash_value.pkl','wb'))




if __name__ == "__main__":
  app.run(debug=True)
