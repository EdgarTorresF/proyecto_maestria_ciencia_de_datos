import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.svm import LinearSVC
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

st.title("Proyecto Final Ciencia de Datos")
st.subheader("Creador: Edgar Jesus Torres Flores")
st.subheader("Maestría en Ciencia de Datos e Inteligencia Artificial")
st.subheader("Fecha: 2 de Diciembre de 2024")

st.divider()
# Cargamos las bases de datos
# @st.cache_data
app_record = pd.read_csv('/Users/edgartorres/Documents/GitHub/Proyecto Final Ciencia de Datos/application_record.csv')
cred_record = pd.read_csv('/Users/edgartorres/Documents/GitHub/Proyecto Final Ciencia de Datos/credit_record.csv')

st.text("Revisamos algunos datos de la base de datos de Solicitudes")
st.dataframe(app_record)

st.text("Revisamos algunos datos de la base de datos de Créitos")
st.dataframe(cred_record)

st.divider()

fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=(
        'Género', 'Ingresos', 'Nivel Educativo', 
        'Estatus Marital', 'Tipo de Hogar', 'Edad', 
        'Antigüedad Laboral'
    )
)

# Distribución por Género
gender_fig = px.histogram(
    app_record, x='CODE_GENDER', color='CODE_GENDER', title='Distribución por Género',
    color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_dark'
)
for trace in gender_fig.data:
    fig.add_trace(trace, row=1, col=1)

# Distribución por Ingresos
income_fig = px.histogram(
    app_record, x='AMT_INCOME_TOTAL', nbins=50, title='Distribución por Ingresos',
    color_discrete_sequence=['#00CC96'], template='plotly_dark'
)
for trace in income_fig.data:
    fig.add_trace(trace, row=1, col=2)

# Distribución por Nivel Educativo
education_fig = px.histogram(
    app_record, x='NAME_EDUCATION_TYPE', color='NAME_EDUCATION_TYPE', 
    title='Distribución por Nivel Educativo',
    color_discrete_sequence=px.colors.qualitative.Set2, template='plotly_dark'
)
for trace in education_fig.data:
    fig.add_trace(trace, row=1, col=3)

# Distribución por Estado Civil
family_fig = px.histogram(
    app_record, x='NAME_FAMILY_STATUS', color='NAME_FAMILY_STATUS', 
    title='Distribución por Estado Civil',
    color_discrete_sequence=px.colors.qualitative.Vivid, template='plotly_dark'
)
for trace in family_fig.data:
    fig.add_trace(trace, row=2, col=1)

# Distribución por Tipo de Hogar
housing_fig = px.histogram(
    app_record, x='NAME_HOUSING_TYPE', color='NAME_HOUSING_TYPE', 
    title='Distribución por Tipo de Hogar',
    color_discrete_sequence=px.colors.qualitative.Alphabet, template='plotly_dark'
)
for trace in housing_fig.data:
    fig.add_trace(trace, row=2, col=2)

# Distribución por Edad
app_record['AGE'] = app_record['DAYS_BIRTH'] // -365
age_fig = px.histogram(
    app_record, x='AGE', nbins=50, title='Distribución por Edad',
    color_discrete_sequence=['#FFA15A'], template='plotly_dark'
)
for trace in age_fig.data:
    fig.add_trace(trace, row=2, col=3)

# Distribución por Antigüedad Laboral
app_record['YEARS_EMPLOYED'] = app_record['DAYS_EMPLOYED'] / 365
employed_fig = px.histogram(
    app_record, x='YEARS_EMPLOYED', nbins=50, title='Distribución por Antigüedad Laboral',
    color_discrete_sequence=['#AB63FA'], template='plotly_dark'
)
for trace in employed_fig.data:
    fig.add_trace(trace, row=3, col=1)

# Ajustes del layout
fig.update_layout(
    height=900, width=1200, title_text='Distribución de Variables', 
    title_font_size=24, title_x=0.5, showlegend=False
)

fig

st.divider()

# Ingreso vs Edad por Género
scatter_fig = px.scatter(app_record, x='AGE', y='AMT_INCOME_TOTAL', color='CODE_GENDER',
                         title='Ingreso vs Edad por Género', color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_dark')
scatter_fig.update_layout(title_font_size=20, title_x=0.5, xaxis_title='Edad', yaxis_title='Ingresos')
scatter_fig

# Ingreso vs Ocupación por Género
scatter_fig = px.scatter(app_record, x='OCCUPATION_TYPE', y='AMT_INCOME_TOTAL', color='CODE_GENDER',
                         title='Ingreso vs Ocupación por Género', color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_dark')
scatter_fig.update_layout(title_font_size=20, title_x=0.5, xaxis_title='Ocupación', yaxis_title='Ingresos')
scatter_fig

# Ingreso vs Nivel Educativo por Género
scatter_fig = px.scatter(app_record, x='NAME_EDUCATION_TYPE', y='AMT_INCOME_TOTAL', color='CODE_GENDER',
                         title='Ingreso vs Nivel Educativo por Género', color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_dark')
scatter_fig.update_layout(title_font_size=20, title_x=0.5, xaxis_title='Nivel Educativo', yaxis_title='Ingresos')
scatter_fig

# Ingreso vs Antigüedad Laboral por Género
scatter_fig = px.scatter(app_record, x='YEARS_EMPLOYED', y='AMT_INCOME_TOTAL', color='CODE_GENDER',
                         title='Ingreso vs Antigüedad Laboral por Género', color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_dark')
scatter_fig.update_layout(title_font_size=20, title_x=0.5, xaxis_title='Nivel Educativo', yaxis_title='Ingresos')
scatter_fig

# Ingreso por Estado Civil
scatter_fig = px.scatter(app_record, x='NAME_FAMILY_STATUS', y='AMT_INCOME_TOTAL', color='NAME_FAMILY_STATUS',
                         title='Ingreso por Estado Civil', color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_dark')
scatter_fig.update_layout(title_font_size=20, title_x=0.5, xaxis_title='Estado Civil', yaxis_title='Ingresos')
scatter_fig

st.divider()

# Heatmap de la Matriz de Correlación
numerical_cols = app_record.select_dtypes(include=[np.number]).columns
correlation_matrix = app_record[numerical_cols].corr()

# Crear Heatmap
heatmap_fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='Viridis',
    colorbar=dict(title='Correlación'),
    zmin=-1, zmax=1, 
    showscale=True  
))

# Agregar etiquetas
annotations = []
for i, row in enumerate(correlation_matrix.values):
    for j, value in enumerate(row):
        annotations.append(
            dict(
                text=str(round(value, 2)),
                x=correlation_matrix.columns[j],
                y=correlation_matrix.columns[i],
                xref='x1', yref='y1',
                font=dict(color='black', size=12),
                showarrow=False
            )
        )

heatmap_fig.update_layout(
    title='Matriz de Correlación',
    xaxis_nticks=36,
    template='plotly_dark',
    annotations=annotations  # Agregar etiquetas
)

heatmap_fig

# Creamos un nuevo dataframe uniendo app_record y cred_record en "ID"
merged_df = pd.merge(app_record, cred_record, on='ID')


# Clasificamos a los clientes como buenos o malos de acuerdo al STATUS y agregamos esa columna al merged_df
def classify_client(status):
    if status in ['2', '3', '4', '5']:
        return 'bad'
    else:
        return 'good'

merged_df['client_status'] = merged_df['STATUS'].apply(classify_client)


# Se utiliza LabelEncoder para normalizar las etiquetas de nuestra columna objetivo
label_encoder = LabelEncoder()
merged_df['client_status'] = label_encoder.fit_transform(merged_df['client_status'])



# Se eliminan las columnas que no se van a utilizar en el mdoelo
merged_df.drop(columns=['ID', 'STATUS', 'MONTHS_BALANCE'], inplace=True)


# Se codifican las variables categoricas
categorical_cols = merged_df.select_dtypes(include=['object']).columns
merged_df = pd.get_dummies(merged_df, columns=categorical_cols, drop_first=True)


# Usamos la media para rellenar valores Nulos
merged_df.fillna(merged_df.median(), inplace=True)

# Se dividen las variables que vamos a usar en el modelo y el objetivo 
X = merged_df.drop(columns=['client_status'])
y = merged_df['client_status']


# Hacemos un balanceo de los datos usando SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)



# Se estandarizan las variables con StandardScaler
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)


# Se hace la division de datos de entrenamiento y datos de prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)




# Se crear la funcio para las Matrices de Confusion de los Modelos
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)



st.divider()
st.subheader("Resultados de un Modelo SVC")
col1, col2 = st.columns(2)

# Modelo de SVC
svc = LinearSVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
precision, recall, f1score, _ = precision_recall_fscore_support(y_test, y_pred_svc, average='micro')
cm_svc = confusion_matrix(y_test, y_pred_svc)
with col1:
    st.metric("Accuracy de SVC: ",accuracy_svc)
    st.metric("Precision de SVC: ",precision)
    st.metric("Recall de SVC: ",recall)
    st.metric("F1 Score de SVC: ",f1score)
with col2:
    plot_confusion_matrix(cm_svc, labels=svc.classes_)


st.divider()
st.subheader("Resultados de un Modelo Random Forest")

col1, col2 = st.columns(2)

# Modelo Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf_clf = accuracy_score(y_test, y_pred_rf)
precision, recall, f1score, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='micro')
cm_rf = confusion_matrix(y_test, y_pred_rf)
class_labels = np.unique(y_test)
with col1:
    st.metric("Accuracy de Random Forest: ",accuracy_rf_clf)
    st.metric("Precision de Random Forest: ",precision)
    st.metric("Recall de Random Forest: ",recall)
    st.metric("F1 Score de Random Forest: ",f1score)
with col2:   
    plot_confusion_matrix(cm_rf, labels=class_labels)

st.divider()
st.subheader("Resultados de un Modelo de Arboles de Decisión")

col1, col2 = st.columns(2)

# Modelo de Arboles de Decisión
decision_tree_clf = DecisionTreeClassifier(max_depth=5)
decision_tree_clf.fit(X_train, y_train)
y_pred_dt = decision_tree_clf.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision, recall, f1score, _ = precision_recall_fscore_support(y_test, y_pred_dt, average='micro')
cm_dt = confusion_matrix(y_test, y_pred_dt)
class_labels = np.unique(y_test)
with col1:
    st.metric("Accuracy de Arboles de Decision: ",accuracy_dt)
    st.metric("Precision de Arboles de Decision: ",precision)
    st.metric("Recall de Arboles de Decision: ",recall)
    st.metric("F1 Score de Arboles de Decision: ",f1score)
with col2:    
    plot_confusion_matrix(cm_dt, labels=class_labels)

st.divider()
best_model = max(
    {"SVC": accuracy_svc, "Random Forest": accuracy_rf_clf, "Decision Tree": accuracy_dt},
    key=lambda x: {"SVC": accuracy_svc, "Random Forest": accuracy_rf_clf, "Decision Tree": accuracy_dt}[x],
)
st.subheader("Conclusión")
st.write(f"El mejor modelo basado en la precisión es **{best_model}** con una precisión de {max(accuracy_svc, accuracy_rf_clf, accuracy_dt):.4f}.")

