import streamlit as st
import pandas as pd 
import plotly_express as px


# title of streamlit app

st.title('Candidate Job Ranking System Using Natural Language Processing (NLP)')

# sidebar

st.sidebar.subheader('Files')
uploaded_file = st.sidebar.file_uploader(label='Please upload your CV', type=['docx'])

print(uploaded_file)


