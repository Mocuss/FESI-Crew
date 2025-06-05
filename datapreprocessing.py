import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

def preprocess_data(customers, geolocation, order_items, order_payments, order_reviews, orders, products, product_category_name_translation):
    print("Preprocessing started...")
    # Load the datasets
    customers_df = pd.read_csv(customers, on_bad_lines='skip')
    geolocation_df = pd.read_csv(geolocation, on_bad_lines='skip')
    order_items_df = pd.read_csv(order_items, on_bad_lines='skip')
    order_payments_df = pd.read_csv(order_payments, on_bad_lines='skip')
    order_reviews_df = pd.read_csv(order_reviews, on_bad_lines='skip')
    orders_df = pd.read_csv(orders, on_bad_lines='skip')
    products_df = pd.read_csv(products, on_bad_lines='skip')
    product_category_name_translation_df = pd.read_csv(product_category_name_translation, on_bad_lines='skip')

    # Data Cleaning and Preprocessing
    # Remove duplicates based on 'review_id' and keeping the first occurrence
    order_reviews_df = order_reviews_df.drop_duplicates(subset='review_id', keep='first')
    # Remove duplicates based on 'order_id' and keeping the first occurrence
    order_reviews_df = order_reviews_df.drop_duplicates(subset='order_id', keep='first')

    # Calculate z-scores for latitude and longitude
    geolocation_df["lat_z"] = zscore(geolocation_df["geolocation_lat"])
    geolocation_df["lng_z"] = zscore(geolocation_df["geolocation_lng"])

    # Set a threshold (e.g. 3 standard deviations from the mean)
    threshold = 10
    # Identify rows where either lat or lng z-score is above the threshold
    outliers = geolocation_df[(geolocation_df["lat_z"].abs() > threshold) | (geolocation_df["lng_z"].abs() > threshold)]
    # Drop the z-score columns if not needed
    geolocation_df.drop(columns=["lat_z", "lng_z"], inplace=True)
    # Removing outliers from the geolocation DataFrame
    geolocation_df = geolocation_df.drop(outliers.index)

    # Handle missing or invalid order status and remove unfilled rows
    delivered_with_nulls = orders_df[(orders_df['order_status'] == 'delivered') & (orders_df.isnull().any(axis=1))]
    orders_df = orders_df.drop(delivered_with_nulls.index)

    # Removing rows with missing product category names
    empty_product_name = products_df[products_df['product_category_name'].isnull()]
    product_ids_to_remove = empty_product_name['product_id']
    def remove_product_ids(df):
        df = df[~df['product_id'].isin(product_ids_to_remove)]
        return df
    remove_product_ids(products_df)

    # Handling product weight and dimensions (removing invalid rows)
    empty = products_df[products_df['product_weight_g'].isnull()]
    product_ids_to_remove = pd.concat([product_ids_to_remove, pd.Series(['09ff539a621711667c43eba6a3bd8466'])], ignore_index=True)
    products_df = remove_product_ids(products_df)
    order_items_df = remove_product_ids(order_items_df)

    # Group by 'order_id' and 'product_id' and get the row with the highest 'order_item_id'
    order_items_df = order_items_df.loc[order_items_df.groupby(['order_id', 'product_id'])['order_item_id'].idxmax()]

    # Rename 'order_item_id' to 'quantity'
    order_items_df = order_items_df.rename(columns={'order_item_id': 'quantity'})

    # Add a new column to identify if the payment_type is 'voucher'
    order_payments_df['is_voucher'] = order_payments_df['payment_type'] == 'voucher'

    # Count the number of vouchers for each 'order_id'
    voucher_counts = order_payments_df[order_payments_df['is_voucher']].groupby('order_id').size().reset_index(name='voucher_count')

    # Perform aggregation and keep 'payment_type' as well
    orderpaymentmerge = order_payments_df.groupby('order_id').agg({
        'payment_value': 'sum',         # Total payment for the order
        'is_voucher': 'any',            # Whether any voucher was used in the order
        'payment_type': 'first'         # Keep the first 'payment_type' for each 'order_id'
    }).reset_index()

    # Rename columns for clarity
    orderpaymentmerge.rename(columns={
        'payment_value': 'total_payment',
        'is_voucher': 'voucher_used'
    }, inplace=True)

    # Merge with voucher counts to get the number of vouchers used per order
    orderpaymentmerge = pd.merge(orderpaymentmerge, voucher_counts, on='order_id', how='left')

    # Fill missing 'voucher_count' values with 0 and convert to integer
    orderpaymentmerge['voucher_count'] = orderpaymentmerge['voucher_count'].fillna(0).astype(int)

    # Final Cleaned and Aggregated Payment Data
    order_payments_df = orderpaymentmerge

    # Merging DataFrames
    customer_orders_df = pd.merge(customers_df, orders_df, on='customer_id', how="inner")
    customer_orders_payment_df = pd.merge(customer_orders_df, order_payments_df, on='order_id', how="inner")

    # Ensure geolocation zip codes match customer data
    filtered_customer_orders_payment_df = customer_orders_payment_df[customer_orders_payment_df['customer_zip_code_prefix'].isin(geolocation_df['geolocation_zip_code_prefix'])]

    # Merge customer, order, payment with reviews
    filtered_customer_orders_payment_reviews_df = pd.merge(filtered_customer_orders_payment_df, order_reviews_df, on='order_id', how="inner")

    # Merge Product category names with products dataset
    products_df = products_df.merge(product_category_name_translation_df, on='product_category_name', how='left')
    products_df['product_category_name'] = products_df['product_category_name_english']
    products_df.drop(columns=['product_category_name_english'], inplace=True)

    # Merging Product category with order items
    products_order_items_df = order_items_df.merge(products_df, on='product_id', how='left')

    # Group products by order_id and aggregate necessary values
    products_order_items_df_grouped = products_order_items_df.groupby('order_id', as_index=False).agg({
        'product_category_name': lambda x: ', '.join(sorted(set(x.dropna()))),
        'price': 'sum',
        'freight_value': 'sum',
        'quantity': 'sum'
    })

    # Merging the final dataset together
    merged_dataset = filtered_customer_orders_payment_reviews_df.merge(products_order_items_df_grouped, on='order_id', how='inner')

    # Ensure datetime columns are properly created and time-based features are added
    merged_dataset['order_purchase_timestamp'] = pd.to_datetime(merged_dataset['order_purchase_timestamp'])
    merged_dataset['order_approved_at'] = pd.to_datetime(merged_dataset['order_approved_at'])
    merged_dataset['order_delivered_carrier_date'] = pd.to_datetime(merged_dataset['order_delivered_carrier_date'])
    merged_dataset['review_creation_date'] = pd.to_datetime(merged_dataset['review_creation_date'])

    # Extract time-based features
    merged_dataset['order_purchase_hour'] = merged_dataset['order_purchase_timestamp'].dt.hour
    merged_dataset['order_purchase_day'] = merged_dataset['order_purchase_timestamp'].dt.day
    merged_dataset['order_purchase_weekday'] = merged_dataset['order_purchase_timestamp'].dt.weekday
    merged_dataset['order_purchase_month'] = merged_dataset['order_purchase_timestamp'].dt.month

    merged_dataset['order_to_approval_time'] = (merged_dataset['order_approved_at'] - merged_dataset['order_purchase_timestamp']).dt.total_seconds()
    merged_dataset['approval_to_delivery_time'] = (merged_dataset['order_delivered_carrier_date'] - merged_dataset['order_approved_at']).dt.total_seconds()
    merged_dataset['delivery_to_review_time'] = (merged_dataset['review_creation_date'] - merged_dataset['order_delivered_carrier_date']).dt.total_seconds()

    # Step 1: Count the number of orders for each customer
    customer_order_counts = merged_dataset.groupby('customer_unique_id')['order_id'].nunique().reset_index()
    customer_order_counts.columns = ['customer_unique_id', 'order_count']

    # Step 2: Merge the order counts with the main dataset to add 'order_count' as a feature
    merged_dataset = pd.merge(merged_dataset, customer_order_counts, on='customer_unique_id', how='left')

    # Step 3: Create the 'repeat_buyer' column: 
    # 1 if order_count > 1 (repeat buyer), else 0 (non-repeat buyer)
    merged_dataset['repeat_buyer'] = np.where(merged_dataset['order_count'] > 1, 1, 0)

    # Check the new target variable
    print(merged_dataset[['customer_unique_id', 'order_count', 'repeat_buyer']].head())

    # Label Encoding for Ordinal Categories
    label_encoder = LabelEncoder()
    merged_dataset['order_status'] = label_encoder.fit_transform(merged_dataset['order_status'])  # Ordinal encoding
    merged_dataset['review_score'] = label_encoder.fit_transform(merged_dataset['review_score'])  # Ordinal encoding

    # One-Hot Encoding for Nominal Categories
    merged_dataset = pd.get_dummies(merged_dataset, columns=['product_category_name', 'customer_state'], drop_first=True)

    # Step 5: Select relevant numeric columns for analysis
    df1 = merged_dataset[['price', 'freight_value', 'quantity', 'order_to_approval_time', 
                        'approval_to_delivery_time', 'delivery_to_review_time', 'order_purchase_hour',
                        'order_purchase_day', 'order_purchase_weekday', 'order_purchase_month', 
                        'order_status', 'review_score', 'repeat_buyer']]

    # Step 6: Handle missing values
    imputer = SimpleImputer(strategy='mean')  # Fill missing values with the column mean
    df1 = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)

    # Step 7: Apply Log Transformation to Skewed Features (price, freight_value, quantity)
    df1['price'] = np.log1p(df1['price'])
    df1['freight_value'] = np.log1p(df1['freight_value'])
    df1['quantity'] = np.log1p(df1['quantity'])

    # Step 8: Apply RobustScaler to handle outliers
    scaler = RobustScaler()
    df1_scaled = scaler.fit_transform(df1)

    # Convert the scaled data back to a DataFrame
    df1_scaled = pd.DataFrame(df1_scaled, columns=df1.columns)

    # The preprocessed and scaled data is now ready for modeling
    print(df1_scaled.head())  # Display the first few rows of the processed data

    return df1_scaled
