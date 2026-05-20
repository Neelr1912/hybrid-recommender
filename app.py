import streamlit as st
import pandas as pd
from data_adapter import adapt_data
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender

st.title("📊 Hybrid Recommender System")

uploaded_file = st.file_uploader("Upload your dataset (CSV)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Raw Data")
    st.dataframe(df.head())

    # Apply Data Adapter
    adapted_df, meta = adapt_data(df)

    st.write("### Detected Columns")
    st.json(meta)

    st.write("### Adapted Data")
    st.dataframe(adapted_df.head())

    # Build models
    content_model = ContentRecommender(adapted_df)

    collab_model = None
    if meta['has_user_data']:
        collab_model = CollaborativeRecommender(adapted_df)

    hybrid_model = HybridRecommender(content_model, collab_model)

    item_list = adapted_df['title'].dropna().unique()
    # Limit selectbox options to prevent browser OOM with huge datasets
    if len(item_list) > 100:
        item_list = item_list[:100]
    selected_item = st.selectbox("Select Item", item_list)
    min_price, max_price = st.slider(
    "Select Price Range",
    min_value=0,
    max_value=5000,
    value=(0, 5000)
)
if st.button("Recommend"):

    recs = hybrid_model.recommend(selected_item)

    # Convert list to DataFrame
    recs_df = pd.DataFrame(recs)
    st.write(recs_df.columns)

    # Apply price filtering
    if 'price' in recs_df.columns:
        recs_df = recs_df[
            (recs_df['price'] >= min_price) &
            (recs_df['price'] <= max_price)
        ]

    st.write("### Recommendations")
    st.dataframe(recs_df)