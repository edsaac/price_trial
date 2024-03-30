import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" """
st.title('Predicting Electricity Prices at the Alberta Region')
st.markdown('This app allows predicting Electricty prices for the Alberta region considering the "Residential", "Commercial" and the "Industrial" Sectors.')
""" """
data = pd.read_csv('df_.csv')
# Get column names
column_names = list(data.columns)[1:-1]

st.sidebar.markdown('<h2 style="color: blue;"> Select the values of input variables to predict the target variable</h2>', unsafe_allow_html=True)
user_input_prediction = {}
for column in column_names:
  #print(column)
  if data[column].dtype != 'O':
    user_input_prediction[column] = st.sidebar.slider(f'{column}', float(data[column].min()), float(data[column].max()), float(data[column].mean()))
""" """
df = pd.DataFrame()
list_ = sorted(data['Sector'].unique().tolist())
#
for sector_ in list_:
  #print(sector_)
  user_input_prediction['Sector'] = list_.index(sector_)
  df = pd.concat([df, pd.DataFrame([user_input_prediction])], axis = 0, ignore_index=True)
df['Sector'] = df['Sector'].astype('float')
""" """
model = joblib.load('lgbm_model.sav')
"""model.set_params(n_classes=2)"""

""" """
preds = model.predict(df.values)
st.subheader('Prediction')
#
if st.sidebar.button("Predict Electricity Prices"):
  dict_ = {}
  for column in column_names:
    if data[column].dtype != 'O':
      dict_[column] = user_input_prediction[column]
    else:
      dict_[column] = list(data['Sector'].unique())
  df_pred = pd.DataFrame(dict_)
  le = LabelEncoder()
  df_pred['Sector'] = le.fit_transform(df_pred['Sector'])
  features_list = df_pred.values.tolist()
  preds = model.predict(features_list)
  """ """
  df_output = pd.DataFrame(np.round(preds,2)).T
  df_output.columns = list_
  st.text(df_output)
  st.bar_chart(df_output)
"""fig, ax=plt.subplots(figsize=(8,5))
colors_ = sns.color_palette("deep")
ax = sns.barplot(df_output, palette=colors_)
for i in range(len(list_)):
  ax.bar_label(ax.containers[i], fontsize=10);
#ax.grid(axis='y')
plt.ylabel('Electricity Price - (CAD Cents/KWh)')
plt.show()
st.pyplot(fig)
"""
