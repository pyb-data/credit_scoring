
import streamlit as st
from pickle import load
import pandas as pd
import plotly.graph_objects as go


imp = load(open('imp.pkl','rb'))
df = load(open('imp.pkl','rb'))
df = load(open('seuil_stat.pkl','rb'))

# Add title to the page.
st.title("Importance")

st.write(imp)

