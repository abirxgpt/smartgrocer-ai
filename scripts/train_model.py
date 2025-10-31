import argparse
import pickle
import json
from datetime import datetime
from scipy.sparse import save_npz, csr_matrix
from src.data_loader import load_instacart_data, create_user_item_matrix
from src.collaborative_filtering import train_collaborative_filtering
from src.content_based import train_content_based_model
from src.recommender import HybridRecommender

def main(data_path, output_dir):
    print("Loading data...")
    data_dict = load_instacart_data(data_path)
    
    print("Creating user-item matrix...")
    user_item_matrix, filtered_df = create_user_item_matrix(
        data_dict['order_products'], min_orders=5
    )
    
    print("Training collaborative filtering model...")
    cf_model, testset, trainset = train_collaborative_filtering(user_item_matrix)
    
    print("Training content-based model...")
    similarity_matrix, vectorizer, products_df = train_content_based_model(
        data_dict['products']
    )
    
    print("Exporting models...")
    
    with open(f'{output_dir}cf_model.pkl', 'wb') as f:
        pickle.dump(cf_model, f)
    
    sparse_similarity_matrix = csr_matrix(similarity_matrix)
    save_npz(f'{output_dir}similarity_matrix.npz', sparse_similarity_matrix)
    
    products_df.to_csv(f'{output_dir}products_enhanced.csv', index=False)
    
    metadata = {
        'train_date': datetime.now().isoformat(),
        'model_version': '1.0',
        'n_products': len(products_df),
        'cf_weight': 0.6,
        'content_weight': 0.4
    }
    
    with open(f'{output_dir}model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Models exported successfully to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/instacart/', help='Path to Instacart data')
    parser.add_argument('--output_dir', default='./models/', help='Directory to save models')
    args = parser.parse_args()
    
    main(args.data_path, args.output_dir)