# SmartGrocer AI - Hybrid Recommendation System

An intelligent product recommendation system for quick commerce platforms, built using collaborative filtering and content-based approaches on the Instacart dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

## Features

- **Hybrid Recommendation Engine**: Combines Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF)
- **High Performance**: Precision@10: 18.08%, Recall@10: 52.26%, F1: 26.87%
- **Interactive Web App**: Built with Streamlit for easy product recommendations
- **Cold Start Handling**: Works for both existing users and new cart-based queries
- **Explainable AI**: Provides reasoning behind each recommendation

## Demo

Try the live demo: [Coming Soon]

## Architecture

```
SmartGrocer
├── Collaborative Filtering (SVD)
│   └── User-Item Matrix → Latent Factor Model
├── Content-Based Filtering (TF-IDF)
│   └── Product Features → Similarity Matrix
└── Hybrid Model
    └── Weighted Combination (60% CF + 40% Content)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (for model loading)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/abirxgpt/smartgrocer-ai.git
cd smartgrocer-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
```bash
# Download from Google Drive
gdown --id 10YdEByyvH_QwkppzvnpUiPLz2fk54qMn
unzip models.zip
```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the Recommendation Engine

```python
from src.recommender import HybridRecommender
import pickle

with open('models/cf_model.pkl', 'rb') as f:
    cf_model = pickle.load(f)

recommender = HybridRecommender(cf_model, similarity_matrix, products_df)

user_id = 12345
user_history = [1234, 5678, 9012]
recommendations = recommender.recommend(user_id, user_history, n_recommendations=10)

print(recommendations['product_name'].tolist())
```

### Training Your Own Model

```bash
python scripts/train_model.py --data_path ./data/instacart/ --output_dir ./models/
```

## Dataset

This project uses the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset:

- 3M+ orders from 200K+ users
- 50K+ products across 21 departments
- 32M+ user-product interactions

## Model Performance

| Metric | Score |
|--------|-------|
| RMSE | 1.2847 |
| Precision@10 | 18.08% |
| Recall@10 | 52.26% |
| F1@10 | 26.87% |

## Project Structure

```
smartgrocer-ai/
│
├── src/
│   ├── data_loader.py
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   └── recommender.py
│
├── scripts/
│   ├── train_model.py
│   └── export_models.py
│
├── models/
│   ├── cf_model.pkl
│   ├── similarity_matrix.npz
│   └── products_enhanced.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

## Tech Stack

- **Machine Learning**: scikit-surprise, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, seaborn, matplotlib
- **Web Framework**: Streamlit
- **Deployment**: Docker (optional)

## API Reference

### HybridRecommender Class

```python
recommender = HybridRecommender(
    cf_model,           # Trained SVD model
    similarity_matrix,  # Product similarity matrix
    products_df,        # Product metadata
    cf_weight=0.6,      # Weight for collaborative filtering
    content_weight=0.4  # Weight for content-based
)
```

**Methods:**

- `recommend(user_id, user_history, n_recommendations=10)`: Generate recommendations
- `explain_recommendation(user_id, product_id, user_history)`: Explain why a product was recommended

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Instacart for providing the dataset
- The scikit-surprise and scikit-learn communities
- All contributors to this project

## Contact

Abir - [@abirxgpt](https://github.com/abirxgpt)

Project Link: [https://github.com/abirxgpt/smartgrocer-ai](https://github.com/abirxgpt/smartgrocer-ai)

## Citation

If you use this project in your research or application, please cite:

```bibtex
@software{smartgrocer2025,
  author = {Abir},
  title = {SmartGrocer AI: Hybrid Recommendation System for Quick Commerce},
  year = {2025},
  url = {https://github.com/abirxgpt/smartgrocer-ai}
}
```
