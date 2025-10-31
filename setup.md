# Setup Guide

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/abirxgpt/smartgrocer-ai.git
cd smartgrocer-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

```bash
mkdir models
cd models
gdown --id 10YdEByyvH_QwkppzvnpUiPLz2fk54qMn
unzip models.zip
cd ..
```

The models directory should contain:
- `cf_model.pkl` (Collaborative Filtering model)
- `similarity_matrix.npz` (Content-based similarity matrix)
- `products_enhanced.csv` (Product metadata)

### 5. Run the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## Training Your Own Model

### 1. Download Instacart Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

Required files:
- orders.csv
- order_products__prior.csv
- order_products__train.csv
- products.csv
- aisles.csv
- departments.csv

### 2. Organize Data

```bash
mkdir -p data/instacart
# Move downloaded files to data/instacart/
```

### 3. Train Models

```bash
python scripts/train_model.py --data_path ./data/instacart/ --output_dir ./models/
```

Training takes approximately 15-20 minutes on a standard laptop.

## Troubleshooting

### Models Not Loading

If you see "Error loading models", ensure:
1. Models directory exists
2. All three model files are present
3. Files are not corrupted

### Memory Issues

If training fails with memory errors:
1. Reduce `min_orders` parameter in data_loader.py
2. Use a machine with at least 8GB RAM
3. Close other applications

### Import Errors

If scikit-surprise fails to install:
```bash
pip install numpy==1.23.5 --force-reinstall
pip install scikit-surprise
```

## Docker Deployment (Optional)

```bash
docker build -t smartgrocer-ai .
docker run -p 8501:8501 smartgrocer-ai
```

## API Usage

```python
from src.recommender import HybridRecommender
import pickle

with open('models/cf_model.pkl', 'rb') as f:
    cf_model = pickle.load(f)

recommender = HybridRecommender(cf_model, similarity_matrix, products_df)
recommendations = recommender.recommend(user_id=12345, user_history=[1,2,3], n_recommendations=10)
```

## Support

For issues, please open a GitHub issue or contact [@abirxgpt](https://github.com/abirxgpt)