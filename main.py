import os
import time
import schedule
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from supabase import create_client, Client
from datetime import datetime, timezone, timedelta


def run_pipeline():
    print("\nPipeline dijalankan pada", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # === Supabase connection ===
    load_dotenv()
    url: str = os.getenv("SUPABASE_URL")
    key: str = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    # === Load trained model ===
    model = tf.keras.models.load_model("two_tower_model.h5")
    print("✅ Model loaded!")

    # === Load PCA models ===
    user_pca = load("pca_user.joblib")
    product_pca = load("pca_product.joblib")

    # === Load data from Supabase ===
    # df_users = pd.DataFrame(supabase.table("automation_users").select("*").eq("year_month", "2024-11-27").execute().data)
    df_users = pd.DataFrame(supabase.table("automation_users").select("*").execute().data)
    df_products = pd.DataFrame(supabase.table("temp_ik_nudgegamification").select("*").execute().data)

    # === Fetch interactions in batches ===
    all_data = []
    batch_size = 1000
    start = 0
    while True:
        response = supabase.table("automation_interactions").select("*").eq("year_month", "2024-11-27").range(start, start + batch_size - 1).execute()
        # response = supabase.table("automation_interactions").select("*").range(start, start + batch_size - 1).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        start += batch_size

    df_interactions = pd.DataFrame(all_data)

    # === Check 'purchased' column ===
    if 'purchased' not in df_interactions.columns:
        raise ValueError("Column 'purchased' not found in interaction data!")

    df_interactions['purchased'] = df_interactions['purchased'].astype(int)

    # === PCA on numeric features ===
    user_numeric = df_users.select_dtypes(include=['int64', 'float64'])
    product_numeric = df_products.select_dtypes(include=['int64', 'float64'])

    user_pca_features = user_pca.transform(user_numeric)
    product_pca_features = product_pca.transform(product_numeric)

    user_pca_df = pd.DataFrame(user_pca_features, columns=[f"user_pca_{i}" for i in range(user_pca_features.shape[1])])
    product_pca_df = pd.DataFrame(product_pca_features, columns=[f"product_pca_{i}" for i in range(product_pca_features.shape[1])])

    df_users_pca = pd.concat([df_users[['user_id']], user_pca_df], axis=1)
    df_products_pca = pd.concat([df_products[['product_id']], product_pca_df], axis=1)

    # === Merge features into interaction data ===
    df_interactions = df_interactions.merge(df_users_pca, on="user_id", how="left")
    df_interactions = df_interactions.merge(df_products_pca, on="product_id", how="left")

    # === Prepare model input ===
    X_user = df_interactions[[col for col in df_interactions.columns if col.startswith("user_pca_")]].values
    X_product = df_interactions[[col for col in df_interactions.columns if col.startswith("product_pca_")]].values

    # === Handle missing values ===
    imputer = SimpleImputer(strategy="mean")
    X_user = imputer.fit_transform(X_user)
    X_product = imputer.fit_transform(X_product)

    # === Predict ===
    print("🔍 Predicting...")
    y_pred_scores = model.predict([X_user, X_product])
    y_pred_binary = (y_pred_scores > 0.5).astype(int)

    # === Add timestamp ===
    wib_time = datetime.now(timezone(timedelta(hours=7)))
    timestamp = wib_time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_column = [timestamp] * len(y_pred_scores)

    # === Results DataFrame ===
    results_df = pd.DataFrame({
        "user_id": df_interactions["user_id"],
        "product_id": df_interactions["product_id"],
        "score": y_pred_scores.flatten(),
        "predicted_class": y_pred_binary.flatten(),
        "timestamp": timestamp_column
    })

    # Merge product info
    results_df = results_df.merge(df_products, on="product_id", how="left")

    # Optional: filter for a specific user
    filtered_results = results_df[
        results_df["user_id"] == "db712ce7-4268-440c-a6be-95c7ab2a69bc"
    ][['user_id', 'product_id', 'product_title', 'score', 'predicted_class', 'risklevel', 'framingeffect', 'lossaversion']].drop_duplicates('product_id')

    # Delete all rows from the table
    supabase.table("temp_ik_resultsnudgetwscore").delete().neq("user_id", "").execute()
    print("Existing records deleted from temp_ik_resultsnudgetwscore.")

    # Upload to Supabase
    records_to_insert = results_df.to_dict(orient="records")
    supabase.table("temp_ik_resultsnudgetwscore").insert(records_to_insert).execute()
    print("🆙 Results inserted to Supabase.")

# Panggil run_pipeline() untuk menjalankan seluruh pipeline
run_pipeline()

# Jadwalkan fungsi run_pipeline() menggunakan schedule
schedule.every(60).seconds.do(run_pipeline)

print("Scheduler berjalan. Menunggu jadwal eksekusi...")

while True:
    schedule.run_pending()
    time.sleep(30)  
