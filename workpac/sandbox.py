import pandas as pd 


# import data to model CV against
workpac_data = pd.read_csv('job_descriptions.csv')

# import job_candidate data for candidate ranking feature
job_candidate_rank_df = pd.read_csv('job_candidate_similarity.csv')
candidates = pd.read_csv('candidates.csv')

# create table with just data needed for this app

job_candidate_rank_df = pd.merge(left=job_candidate_rank_df, right=workpac_data, left_on='job_link', right_on='job_link')
job_candidate_rank_df = pd.merge(left=job_candidate_rank_df, right=candidates, left_on='candidate_id', right_on='candidate_id')


job_candidate_rank_df = job_candidate_rank_df[['job_title', 'job_link', 'candidate_id', 'candidate_name', 'cosine_similarity']]
