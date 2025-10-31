import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
import plotly.graph_objects as go

st.set_page_config(page_title="SmartGrocer AI", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.title {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin: 8px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric {
    background: #f0f2f6;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    with open("models/cf_model.pkl", "rb") as f:
        cf_model = pickle.load(f)
    similarity_matrix = load_npz("models/similarity_matrix.npz").toarray()
    products = pd.read_csv("models/products_enhanced.csv")
    return cf_model, similarity_matrix, products

try:
    cf_model, sim_matrix, products_df = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

st.markdown('<div class="title">🛒 SmartGrocer AI</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric"><b>Precision@10</b><br>18.08%</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric"><b>Recall@10</b><br>52.26%</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric"><b>F1 Score</b><br>26.87%</div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric"><b>Products</b><br>{len(products_df):,}</div>', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2 = st.tabs(["🎲 Random User Demo", "🛒 Build Your Cart"])

with tab1:
    st.markdown("### Generate a random user and see personalized recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🎲 Generate Random User", type="primary", use_container_width=True):
            user_id = np.random.randint(10000, 99999)
            n_history = np.random.randint(3, 10)
            history = np.random.choice(products_df["product_id"].values, n_history, replace=False).tolist()
            
            st.session_state.demo_user_id = user_id
            st.session_state.demo_history = history
        
        if "demo_user_id" in st.session_state:
            st.markdown(f"#### 👤 User #{st.session_state.demo_user_id}")
            st.markdown("**Purchase History:**")
            
            for prod_id in st.session_state.demo_history[:6]:
                prod = products_df[products_df["product_id"] == prod_id].iloc[0]
                st.markdown(f"• **{prod['product_name']}**  \n  _{prod['department']}_")
            
            if len(st.session_state.demo_history) > 6:
                st.markdown(f"_...and {len(st.session_state.demo_history)-6} more items_")
    
    with col2:
        if "demo_user_id" in st.session_state:
            with st.spinner("🔮 Generating recommendations..."):
                user_id = st.session_state.demo_user_id
                history_set = set(st.session_state.demo_history)
                
                predictions = []
                for product_id in products_df["product_id"].unique():
                    if product_id not in history_set:
                        pred = cf_model.predict(user_id, product_id)
                        predictions.append((product_id, pred.est))
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                top_10 = predictions[:10]
                
                st.markdown("### 💡 Top 10 Recommendations")
                
                for rank, (prod_id, score) in enumerate(top_10, 1):
                    prod = products_df[products_df["product_id"] == prod_id].iloc[0]
                    
                    st.markdown(f"""
                    <div class="card">
                        <h4>#{rank} {prod['product_name']}</h4>
                        <p><b>Category:</b> {prod['department']} → {prod['aisle']}</p>
                        <p><b>Confidence Score:</b> {score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Click 'Generate Random User' to see recommendations")

with tab2:
    st.markdown("### Add products to your cart and get personalized recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🛒 Your Cart")
        
        all_categories = ["All"] + sorted(products_df["department"].dropna().unique().tolist())
        selected_category = st.selectbox("Filter by category:", all_categories)
        
        if selected_category == "All":
            available_products = products_df["product_name"].tolist()[:300]
        else:
            filtered = products_df[products_df["department"] == selected_category]
            available_products = filtered["product_name"].tolist()[:300]
        
        cart_items = st.multiselect(
            "Add products:",
            available_products,
            help="Select multiple products"
        )
        
        if cart_items:
            st.markdown(f"**Items in cart: {len(cart_items)}**")
            for item in cart_items:
                st.markdown(f"• {item}")
    
    with col2:
        if cart_items:
            if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("🔮 Analyzing your cart..."):
                    cart_ids = products_df[products_df["product_name"].isin(cart_items)]["product_id"].tolist()
                    cart_set = set(cart_ids)
                    
                    predictions = []
                    dummy_user = 999999
                    
                    for product_id in products_df["product_id"].unique():
                        if product_id not in cart_set:
                            pred = cf_model.predict(dummy_user, product_id)
                            predictions.append((product_id, pred.est))
                    
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    top_recs = predictions[:10]
                    
                    st.markdown("### 🎯 You Might Also Like")
                    
                    for rank, (prod_id, score) in enumerate(top_recs, 1):
                        prod = products_df[products_df["product_id"] == prod_id].iloc[0]
                        
                        with st.expander(f"#{rank} {prod['product_name']} (Score: {score:.2f})"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**Department:** {prod['department']}")
                                st.markdown(f"**Aisle:** {prod['aisle']}")
                            with col_b:
                                st.markdown(f"**Confidence:** {score:.2f}")
                                st.markdown(f"**Product ID:** {prod['product_id']}")
        else:
            st.info("👈 Add products to your cart to get recommendations")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><b>🤖 SmartGrocer AI | Hybrid Recommendation Engine</b></p>
    <p>Collaborative Filtering (SVD) + Content-Based Filtering (TF-IDF)</p>
    <p style="font-size: 0.9em;">Built for Quick Commerce Apps | Production-Ready System</p>
</div>
""", unsafe_allow_html=True)