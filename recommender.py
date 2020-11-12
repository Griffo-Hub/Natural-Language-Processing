from text_processing import _removeNonAscii
from text_processing import make_lower_case
from text_processing import remove_stop_words
from text_processing import remove_punctuation
from text_processing import remove_html
from text_processing import stem_sentences
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def recommender(model_df, cv):
    
    # get inputs for recommender system

    key_words = cv
    
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
    
    model_df = 0
    
    return recommended_jobs
