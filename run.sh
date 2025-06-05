#!/bin/bash

# Read the paths from config.yaml using yq
CUSTOMERS_PATH=$(yq eval '.data_sources.customers' config.yaml)
GEOLOCATION_PATH=$(yq eval '.data_sources.geolocation' config.yaml)
ORDER_ITEMS_PATH=$(yq eval '.data_sources.order_items' config.yaml)
ORDER_PAYMENTS_PATH=$(yq eval '.data_sources.order_payments' config.yaml)
ORDER_REVIEWS_PATH=$(yq eval '.data_sources.order_reviews' config.yaml)
ORDERS_PATH=$(yq eval '.data_sources.orders' config.yaml)
PRODUCTS_PATH=$(yq eval '.data_sources.products' config.yaml)
PRODUCT_CATEGORY_NAME_TRANSLATION_PATH=$(yq eval '.data_sources.product_category_name_translation' config.yaml)

MODEL_PATH=$(yq eval '.model.path' config.yaml)

# Step 1: Preprocess the data and create the model
echo "Starting data preprocessing and model training..."

python dataprep/datapreprocessing.py \
  --customers "$CUSTOMERS_PATH" \
  --geolocation "$GEOLOCATION_PATH" \
  --order_items "$ORDER_ITEMS_PATH" \
  --order_payments "$ORDER_PAYMENTS_PATH" \
  --order_reviews "$ORDER_REVIEWS_PATH" \
  --orders "$ORDERS_PATH" \
  --products "$PRODUCTS_PATH" \
  --product_category_name_translation "$PRODUCT_CATEGORY_NAME_TRANSLATION_PATH"

# Step 2: Save the trained model (if needed) and prepare for prediction
echo "Model training complete. Saving the model..."

# Save the model if necessary (this would be in a separate script)
# python save_model.py --model_path "$MODEL_PATH"

# Step 3: Make predictions
echo "Making predictions using the trained model..."

python model/prediction.py \
  --customers "$CUSTOMERS_PATH" \
  --geolocation "$GEOLOCATION_PATH" \
  --order_items "$ORDER_ITEMS_PATH" \
  --order_payments "$ORDER_PAYMENTS_PATH" \
  --order_reviews "$ORDER_REVIEWS_PATH" \
  --orders "$ORDERS_PATH" \
  --products "$PRODUCTS_PATH" \
  --product_category_name_translation "$PRODUCT_CATEGORY_NAME_TRANSLATION_PATH" \
  --model "$MODEL_PATH"

echo "Prediction process complete!"
