# Add the path to the `dataprep` folder to the sys.path for importing
import sys
import os
# Dynamically add the full path to the src/dataprep folder
##THIS PORTION IS USED FOR TESTING TRAIN.PY
current_dir = os.path.dirname(os.path.abspath(__file__))
dataprep_path = os.path.join(current_dir, '..', 'dataprep')
sys.path.append(os.path.abspath(dataprep_path))

from datapreprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  
##THIS IS THE ACTUAL LINE WE USING FOR RUN.SH
# from src.dataprep.datapreprocessing import preprocess_data

# Define the file paths
# Updated file paths based on your new folder structure
customers = "/home/sam/gdrive/data/olist_customers_dataset.csv"
geolocation = "/home/sam/gdrive/data/olist_geolocation_dataset.csv"
order_items = "/home/sam/gdrive/data/olist_order_items_dataset.csv"
order_payments = "/home/sam/gdrive/data/olist_order_payments_dataset.csv"
order_reviews = "/home/sam/gdrive/data/olist_order_reviews_dataset.csv"
orders = "/home/sam/gdrive/data/olist_orders_dataset.csv"
products = "/home/sam/gdrive/data/olist_products_dataset.csv"
product_category_name_translation = "/home/sam/gdrive/data/product_category_name_translation.csv"

# # defining the file paths for the datasets
# customers = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_customers_dataset.csv"
# geolocation = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_geolocation_dataset.csv"
# order_items = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_order_items_dataset.csv"
# order_payments = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_order_payments_dataset.csv"
# order_reviews = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_order_reviews_dataset.csv"
# orders = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_orders_dataset.csv"
# products = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_products_dataset.csv"
# sellers = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/olist_sellers_dataset.csv"
# product_category_name_translation = "C:/Users/Marcus/Documents/NYP/NYPY3/Y3S1/EGT309 AI Solution Development/data/product_category_name_translation.csv"

# Preprocess the data
df1_scaled = preprocess_data(
    customers, geolocation, order_items, order_payments, order_reviews, orders, products, product_category_name_translation
)
#Do the traintest split again
X = df1_scaled.drop(columns=["repeat_buyer"])
y = df1_scaled["repeat_buyer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Load in our saved models
model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
rf_path = os.path.join(model_dir, "rf_model.pkl")
xgb_path = os.path.join(model_dir, "xg_model.pkl")
rf_model = joblib.load(rf_path)
xgb_model = joblib.load(xgb_path)

# === Predict & Evaluate ===
print("Random Forest Results:")
rf_preds = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

print("\n XGBoost Results:")
xgb_preds = xgb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(classification_report(y_test, xgb_preds))
