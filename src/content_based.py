import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_content_based_model(products_df):
    products_df['combined_text'] = (
        products_df['product_name'].fillna('') + ' ' +
        products_df['aisle'].fillna('') + ' ' +
        products_df['department'].fillna('')
    )
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(products_df['combined_text'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return similarity_matrix, vectorizer, products_df

def get_content_recommendations(product_id, similarity_matrix, products_df, n_recommendations=10):
    try:
        idx = products_df[products_df['product_id'] == product_id].index[0]
    except IndexError:
        return pd.DataFrame()
    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    
    product_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    recommendations = products_df.iloc[product_indices].copy()
    recommendations['content_score'] = similarity_scores
    
    return recommendations[['product_id', 'product_name', 'aisle', 'department', 'content_score']]