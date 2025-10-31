import numpy as np
from src.collaborative_filtering import get_cf_recommendations
from src.content_based import get_content_recommendations

class HybridRecommender:
    def __init__(self, cf_model, similarity_matrix, products_df, cf_weight=0.6, content_weight=0.4):
        self.cf_model = cf_model
        self.similarity_matrix = similarity_matrix
        self.products_df = products_df
        self.cf_weight = cf_weight
        self.content_weight = content_weight

    def recommend(self, user_id, user_history=None, n_recommendations=10):
        if user_history is None:
            user_history = []
        
        purchased_set = set(user_history)
        
        cf_recs = get_cf_recommendations(
            self.cf_model, user_id, self.products_df,
            n_recommendations=50, purchased_products=purchased_set
        )
        
        if len(cf_recs) > 0:
            cf_recs['cf_score_norm'] = (
                (cf_recs['cf_score'] - cf_recs['cf_score'].min()) /
                (cf_recs['cf_score'].max() - cf_recs['cf_score'].min() + 1e-10)
            )
        
        content_scores = []
        for _, row in cf_recs.iterrows():
            if len(user_history) > 0:
                similarities = []
                for hist_prod in user_history[-5:]:
                    content_rec = get_content_recommendations(
                        hist_prod, self.similarity_matrix,
                        self.products_df, n_recommendations=100
                    )
                    
                    match = content_rec[content_rec['product_id'] == row['product_id']]
                    if len(match) > 0:
                        similarities.append(match['content_score'].values[0])
                
                content_scores.append(np.mean(similarities) if similarities else 0)
            else:
                content_scores.append(0)
        
        cf_recs['content_score_norm'] = content_scores
        
        if cf_recs['content_score_norm'].max() > 0:
            cf_recs['content_score_norm'] = (
                cf_recs['content_score_norm'] / cf_recs['content_score_norm'].max()
            )
        
        cf_recs['hybrid_score'] = (
            self.cf_weight * cf_recs['cf_score_norm'] +
            self.content_weight * cf_recs['content_score_norm']
        )
        
        cf_recs = cf_recs.sort_values('hybrid_score', ascending=False)
        
        return cf_recs.head(n_recommendations)

    def explain_recommendation(self, user_id, product_id, user_history=None):
        if user_history is None:
            user_history = []
        
        product_name = self.products_df[
            self.products_df['product_id'] == product_id
        ]['product_name'].values[0]
        
        explanation = f"Product: {product_name}\n\n"
        
        pred = self.cf_model.predict(user_id, product_id)
        explanation += f"Similar users often buy this (Score: {pred.est:.2f})\n"
        
        if len(user_history) > 0:
            recent_product = user_history[-1]
            recent_name = self.products_df[
                self.products_df['product_id'] == recent_product
            ]['product_name'].values[0]
            explanation += f"Similar to your recent purchase: {recent_name}\n"
        
        return explanation