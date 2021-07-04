
import altair as alt
from pickle import load
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

#from urllib.error import URLError





#df = load(open('dfApplicationDash.pkl','rb'))
#seuils = load(open('seuil_stat.pkl','rb'))

seuil_stat_col = load(open('seuil_stat_col.pkl','rb'))
seuil_stat_value = load(open('seuil_stat_value.pkl','rb'))
dfApplicationDash_col = load(open('dfApplicationDash_col.pkl','rb'))
dfApplicationDash_value = load(open('dfApplicationDash_value.pkl','rb'))

df = pd.DataFrame(dfApplicationDash_value, columns = dfApplicationDash_col)
seuils = pd.DataFrame(seuil_stat_value, columns = seuil_stat_col)

seuils['score'] = seuils['seuil']
seuils['pourcentage de défault'] = seuils['prct_default_seuil']


st.title("Scoring probabilité de défaut")
st.markdown("Ce dashboard va vous aider à prendre une décision")


seuil_optimal = seuils[seuils.prct_default_seuil >=0.085].tail(1).seuil.values[0] * 100
seuil_optimal = 20.200000000000003


st.sidebar.title("Sélection du client")
st.sidebar.markdown("Choisissez l'identifiant du client:")
chart_visual = st.sidebar.selectbox('SK_ID_CURR',df.SK_ID_CURR)



#title1 = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Niveau de risque de défaut</p>'
title1 = '<p style="font-size: 30px;">Niveau de risque de défaut</p>'
st.markdown(title1, unsafe_allow_html=True)

seuil = st.slider("Selectionner le seuil de score (x100)", min_value=0.0, max_value=100.0, value=seuil_optimal, step=0.1)

prct_default_seuil = seuils[seuils.seuil * 100 <= seuil].sort_values('seuil', ascending=False).head(1).prct_default_seuil.values[0]


selected = pd.DataFrame({'seuil':[seuil/100], 'prct_default_seuil':[prct_default_seuil]})

if prct_default_seuil >= 0.10:
	approche = 'agressive'
elif prct_default_seuil >= 0.07:
	approche = 'modérée'
else:
	approche = 'conservative'
st.write('### L\'approche est ', approche)

seuils4graph = seuils
seuils4graph['ligne'] = seuils4graph.apply(lambda x: 'courbe_seuil_prct', axis=1)

tmp = pd.concat([seuils.seuil, pd.Series([prct_default_seuil for x in list(seuils.seuil)]),seuils.seuil, pd.Series([prct_default_seuil for x in list(seuils.seuil)]),pd.Series(['selected' for x in list(seuils.seuil)])], axis=1)
tmp.columns = seuils4graph.columns
seuils4graph = pd.concat([seuils4graph,tmp])

basic_chart = alt.Chart(seuils4graph).mark_line().encode(
    x=alt.X('score'),
    y=alt.Y('pourcentage de défault'),
    color='ligne'
    # legend=alt.Legend(title='Animals by year')
).properties(
    width=750,
    height=300
).configure_axis(
    labelFontSize=9,
    titleFontSize=10
)
st.altair_chart(basic_chart)



score = df[df.SK_ID_CURR == chart_visual]['SCORE'].values[0]
if score > seuil / 100:
	risk = 'Risque de défaut'
else:
	risk = 'Pas de risque selon le seuil sélectionné'


title2 = '<p style="font-size: 30px;">Risque du client</p>'
st.markdown(title2, unsafe_allow_html=True)

st.write("### Score du client (x100)", score * 100)
st.write('### ', risk)



data = df[df.SK_ID_CURR == chart_visual]
st.write("### Informations client", data)

fig = go.Figure()
