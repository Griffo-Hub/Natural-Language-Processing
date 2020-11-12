import candidate_job_recommender
import job_candidate_rank
import streamlit as st

PAGES = {
    'Job Reccommender System' : candidate_job_recommender,
    'Candidate Recommender System': job_candidate_rank
}

st.sidebar.title('Recommender Type')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()