import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt



st.header("Analisis de Startups")

st.divider()

df = pd.read_csv(r"C:\Users\edgar\Documents\Python Projects\proyecto_maestria_ciencia_de_datos\startup_dataset.csv")
st.dataframe(df.head(10))

df.info()

st.divider()
st.text("Revisamos el porcentaje de valores nulos")
null_values = df.isna().sum()/df.count()*100
null_values

st.text("Graficamos los valores nulos")
fig = plt.figure(figsize=(12,8))
sns.heatmap(df.isna())
fig

st.divider()


