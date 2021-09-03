
import altair as alt
from pickle import load
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
#from urllib.error import URLError



# RECUPERATION DES DONNEES
seuil_stat_col = load(open('pickle/seuil_stat_col.pkl','rb'))
seuil_stat_value = load(open('pickle/seuil_stat_value.pkl','rb'))
seuil_real_pred_stat_col = load(open('pickle/seuil_real_pred_stat_col.pkl','rb'))
seuil_real_pred_stat_value = load(open('pickle/seuil_real_pred_stat_value.pkl','rb'))
dfApplicationDash_col = load(open('pickle/dfApplicationDash_col.pkl','rb'))
dfApplicationDash_value = load(open('pickle/dfApplicationDash_value.pkl','rb'))

ratio_num_col = load(open('pickle/RatioNum_col.pkl','rb'))
ratio_num_value = load(open('pickle/RatioNum_value.pkl','rb'))

ratio_pos = load(open('pickle/ratio_pos.pkl','rb'))
threshold = float(load(open('pickle/threshold.pkl','rb'))) * 100


df = pd.DataFrame(dfApplicationDash_value, columns = dfApplicationDash_col)
seuils = pd.DataFrame(seuil_stat_value, columns = seuil_stat_col)
seuils_real_pred = pd.DataFrame(seuil_real_pred_stat_value, columns = seuil_real_pred_stat_col)
ratio_num = pd.DataFrame(ratio_num_value, columns = ratio_num_col)




# TITRE DU DASHBOARD
st.title("Scoring probabilité de défaut")
st.markdown("Ce dashboard va vous aider à prendre une décision")


 

# SELECTION DU CLIENT
st.sidebar.title("Sélection du client")
st.sidebar.markdown("Choisissez l'identifiant du client (exemples avec risque de défaut:100751,100568,100278,100754,100753):")
chart_visual = st.sidebar.selectbox('SK_ID_CURR',df.SK_ID_CURR)


# SELECTION DU SEUIL DE RISQUE
#title1 = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Niveau de risque de défaut</p>'
title1 = '<p style="font-size: 30px;">Seuil de risque de défaut</p>'
st.markdown(title1, unsafe_allow_html=True)
seuil = st.slider("Selectionner le seuil de score (x100) au delà duquel le client sera considéré comme à risque", min_value=0.0, max_value=100.0, value=threshold, step=0.1)
prct_default_seuil = seuils[seuils.seuil * 100 <= seuil].sort_values('seuil', ascending=False).head(1).prct_default_seuil.values[0]
prct_default_seuil_real_pred = seuils_real_pred[seuils_real_pred.seuil * 100 <= seuil].sort_values('seuil', ascending=False).head(1).prct_default_seuil_real_pred.values[0]

#selected = pd.DataFrame({'seuil':[seuil/100], 'prct_default_seuil':[prct_default_seuil]})

if prct_default_seuil >= ratio_pos + 0.015:
    approche = 'agressive, elle donne plus de clients à risque que réellement'
elif prct_default_seuil >= ratio_pos - 0.015:
    approche = 'modérée, elle donne autant de clients à risque que réellement'
else:
    approche = 'risquée pour la banque, elle donne moins de clients à risque que réellement'
st.write('### L\'approche est ', approche)



# AFFICHAGE DU GRAPHIQUE PRCT DEFAUT EN FONCTION DU SEUIL
st.markdown("Cette courbe donne le pourcentage de personnes prédites à risque en fonction du seuil sélectionné")
seuils4graph = seuils
seuils4graph['ligne'] = seuils4graph.apply(lambda x: 'courbe_seuil_prct', axis=1)

tmp = pd.concat([seuils.seuil, pd.Series([prct_default_seuil for x in list(seuils.seuil)]),pd.Series(['selected' for x in list(seuils.seuil)])], axis=1)
tmp.columns = seuils4graph.columns
tmp2 = pd.concat([seuils.seuil, pd.Series([ratio_pos for x in list(seuils.seuil)]),pd.Series(['real' for x in list(seuils.seuil)])], axis=1)
tmp2.columns = seuils4graph.columns
seuils4graph = pd.concat([seuils4graph,tmp,tmp2])

basic_chart = alt.Chart(seuils4graph).mark_line().encode(
    x=alt.X('seuil'),
    y=alt.Y('prct_default_seuil'),
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


# AFFICHAGE DU GRAPHIQUE PRCT DEFAUT CORRECTEMENT PREDIT EN FONCTION DU SEUIL
st.markdown("Cette courbe donne le pourcentage de personnes réellement à risque qui sont prédites à risque en fonction du seuil sélectionné")
seuils4graph = seuils_real_pred
seuils4graph['ligne'] = seuils4graph.apply(lambda x: 'courbe_seuil_prct_real_pred', axis=1)

tmp = pd.concat([seuils.seuil, pd.Series([prct_default_seuil_real_pred for x in list(seuils_real_pred.seuil)]),pd.Series(['selected' for x in list(seuils.seuil)])], axis=1)
tmp.columns = seuils4graph.columns
seuils4graph = pd.concat([seuils4graph,tmp])

basic_chart = alt.Chart(seuils4graph).mark_line().encode(
    x=alt.X('seuil'),
    y=alt.Y('prct_default_seuil_real_pred'),
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



# AFFICHAGE DU NIVEAU DE RISQUE DU CLIENT
title2 = '<p style="font-size: 30px;">Risque du client</p>'
st.markdown(title2, unsafe_allow_html=True)

score = df[df.SK_ID_CURR == chart_visual]['SCORE'].values[0]
if score > seuil / 100:
    risk = 'Risque de défaut selon le seuil sélectionné'
else:
    risk = 'Pas de risque selon le seuil sélectionné'    
st.write("### Score du client (x100)", score * 100)    
st.write('### ', risk)


# AFFICHAGE DES DONNEES DU CLIENT
st.write('')
st.write('')
title3 = '<p style="font-size: 30px;">Informations du client</p>'
st.markdown(title3, unsafe_allow_html=True)
st.markdown("Les informations du client sont ordonnées selon leur importance dans le scoring")
data = df[df.SK_ID_CURR == chart_visual]
st.write(data)






# AFFICHAGE VARIABLE / RATIO
title3 = '<p style="font-size: 30px;">Influence des variables sur le risque de défaut</p>'
st.write('')
st.write('')
st.markdown(title3, unsafe_allow_html=True)

features = list(df.columns)[2:]


st.markdown("Sur les graphiques suivants, si la droite verticale orange coupe la courbe bleue au dessus de la droite horizontale rouge, cela indique que le client est à risque selon cette variable")


for feature in features[:11]:
    distrib_default = ratio_num[ratio_num.feature == feature].sort_values('borne_inf')
    first = distrib_default.head(1).copy()
    first['borne_sup'] = first['borne_inf']
    last = distrib_default.tail(1).copy()
    last['borne_inf'] = last['borne_sup']
    distrib_default = pd.concat([first, distrib_default, last])
    distrib_default['value'] = (distrib_default['borne_inf'] + distrib_default['borne_sup']) / 2
    distrib_default = distrib_default[['value','ratio']]
    distrib_default['ligne'] = '% de défaut en fonction de la variable'
    distrib_default = distrib_default.reset_index(drop=True)

    tmp = pd.concat([distrib_default.value, pd.Series([ratio_pos for x in list(distrib_default.value)]),pd.Series(['default average' for x in list(distrib_default.value)])], axis=1)
    tmp.columns = distrib_default.columns

    val1 = df[df.SK_ID_CURR == chart_visual][feature].values[0]
    tmp2 = pd.concat([pd.Series([val1 for x in list(distrib_default.ratio)]), distrib_default.ratio, pd.Series(['client value' for x in list(distrib_default.value)])], axis=1)
    tmp2.columns = distrib_default.columns

    distrib_default = pd.concat([distrib_default,tmp, tmp2])

    basic_chart = alt.Chart(distrib_default).mark_line().encode(
        x=alt.X('value'),
        y=alt.Y('ratio'),
        color='ligne'
        # legend=alt.Legend(title='Animals by year')
    ).properties(
        width=750,
        height=300
    ).configure_axis(
        labelFontSize=9,
        titleFontSize=10
    )
    st.write(feature, val1)
    st.altair_chart(basic_chart)







fig = go.Figure()
