
import streamlit as st
from pickle import load
import pandas as pd
import plotly.graph_objects as go


seuil_stat_col = load(open('seuil_stat_col.pkl','rb'))
seuil_stat_value = load(open('seuil_stat_value.pkl','rb'))
dfApplicationDash_col = load(open('dfApplicationDash_col.pkl','rb'))
dfApplicationDash_value = load(open('dfApplicationDash_value.pkl','rb'))

seuil_stat = pd.DataFrame(seuil_stat_value, columns = seuil_stat_col)
dfApplicationDash = pd.DataFrame(dfApplicationDash_value, columns = dfApplicationDash_col)

# Add title to the page.
st.title("Importance")

st.write(seuil_stat)

