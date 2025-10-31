import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_collaborative_filtering(user_item_matrix, test_size=0.2):
    reader = Reader(rating_scale=(1, user_item_matrix['purchase_count'].max()))
    data = Dataset.load_from_df(
        user_item_matrix[['user_id', 'product_id', 'purchase_count']],
        reader
    )
    
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)
    
    return model, testset, trainset

def get_cf_recommendations(model, user_id, products_df, n_recommendations=10, purchased_products=None):
    if purchased_products is None:
        purchased_products = set()
    
    all_products = products_df['product_id'].unique()
    predictions = []
    
    for product_id in all_products:
        if product_id not in purchased_products:
            pred = model.predict(user_id, product_id)
            predictions.append((product_id, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:n_recommendations]
    
    recommendations = pd.DataFrame(top_predictions, columns=['product_id', 'cf_score'])
    recommendations = recommendations.merge(
        products_df[['product_id', 'product_name', 'aisle', 'department']],
        on='product_id'
    )
    
    return recommendations