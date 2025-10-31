from .data_loader import load_instacart_data, create_user_item_matrix
from .collaborative_filtering import train_collaborative_filtering, get_cf_recommendations
from .content_based import train_content_based_model, get_content_recommendations
from .recommender import HybridRecommender

__version__ = "1.0.0"
__all__ = [
    'load_instacart_data',
    'create_user_item_matrix',
    'train_collaborative_filtering',
    'get_cf_recommendations',
    'train_content_based_model',
    'get_content_recommendations',
    'HybridRecommender'
]