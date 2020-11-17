from text_processing import _removeNonAscii
from text_processing import make_lower_case
from text_processing import remove_stop_words
from text_processing import remove_punctuation
from text_processing import remove_html
from text_processing import stem_sentences
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


# import data to model CV against
workpac_data = pd.read_csv('job_descriptions.csv')


# remove id column as we can use index
workpac_data = workpac_data[['job_link','job_description', 'job_title']]

# clean Na's so functions can run
workpac_data = workpac_data.replace(np.nan, ' ', regex=True)

# clean job description
workpac_data['job_description_cleaned'] = workpac_data['job_description'].apply(_removeNonAscii)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func = make_lower_case)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func = remove_stop_words)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func=remove_punctuation)
workpac_data['job_description_cleaned'] = workpac_data.job_description_cleaned.apply(func=remove_html)

workpac_data['job_title_cleaned'] = workpac_data['job_title'].apply(_removeNonAscii)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func = make_lower_case)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func = remove_stop_words)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func=remove_punctuation)
workpac_data['job_title_cleaned'] = workpac_data.job_title_cleaned.apply(func=remove_html)


# stem cleaned data
workpac_data['job_description_cleaned'] = workpac_data['job_description_cleaned'].apply(stem_sentences)
workpac_data['job_title_cleaned'] = workpac_data['job_title_cleaned'].apply(stem_sentences)

workpac_data['job_text_cleaned'] = workpac_data['job_title_cleaned'] + ' ' + workpac_data['job_description_cleaned']

model_df = workpac_data[['job_link', 'job_text_cleaned']]


# get inputs for recommender system

key_words = input()

# clean input
key_words = _removeNonAscii(key_words)
key_words = make_lower_case(key_words)
key_words = remove_stop_words(key_words)
key_words = remove_punctuation(key_words)
key_words = remove_html(key_words)

# stem input

key_words = stem_sentences(key_words)

# add input to the model_df to calculate tfidf of input against existing data
model_df.loc[-1] = key_words
model_df.index = model_df.index + 1
model_df = model_df.sort_index()

# construct tf vectoriser
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words='english')

# fit tf
tfidf_matrix = tf.fit_transform(model_df['job_text_cleaned'])


# recommender
a1 = tfidf_matrix.getrow(0).toarray()
dist_list = []
for i in range(tfidf_matrix.shape[0]):
    a2 = tfidf_matrix.getrow(i).toarray()
    dist = cosine_similarity(a1, a2)
    dist_list.append(dist)

model_df['cosine_similarity'] = dist_list

recommended_jobs = model_df.sort_values('cosine_similarity', ascending=False)

recommended_jobs = recommended_jobs[['job_link','job_text_cleaned', 'cosine_similarity']][1:51]



###############################
transformed_documents_as_array = tfidf_matrix[8].toarray()
# use this line of code to verify that the numpy array represents the same number of documents that we have in the file list
len(transformed_documents_as_array)


# loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
for counter, doc in enumerate(transformed_documents_as_array):
    # construct a dataframe
    tf_idf_tuples2 = list(zip(tf.get_feature_names(), doc))
    one_doc_as_df2 = pd.DataFrame.from_records(tf_idf_tuples2, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)
