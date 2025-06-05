# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import contextily as ctx

# defining the file paths for the datasets
customers = "X:/data/olist_customers_dataset.csv"
geolocation = "X:/data/olist_geolocation_dataset.csv"
order_items = "X:/data/olist_order_items_dataset.csv"
order_payments = "X:/data/olist_order_payments_dataset.csv"
order_reviews = "X:/data/olist_order_reviews_dataset.csv"
orders = "X:/data/olist_orders_dataset.csv"
products = "X:/data/olist_products_dataset.csv"
sellers = "X:/data/olist_sellers_dataset.csv"
product_category_name_translation = "X:/data/product_category_name_translation.csv"

customers_df = pd.read_csv(customers, on_bad_lines='skip')
geolocation_df = pd.read_csv(geolocation, on_bad_lines='skip')
order_items_df = pd.read_csv(order_items, on_bad_lines='skip')
order_payments_df = pd.read_csv(order_payments, on_bad_lines='skip')
order_reviews_df = pd.read_csv(order_reviews, on_bad_lines='skip')
orders_df = pd.read_csv(orders, on_bad_lines='skip')
products_df = pd.read_csv(products, on_bad_lines='skip')
sellers_df = pd.read_csv(sellers, on_bad_lines='skip')
product_category_name_translation_df = pd.read_csv(product_category_name_translation, on_bad_lines='skip')

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

# removing outliers from the geolocation DataFrame
geolocation_df = geolocation_df.drop(outliers.index)

# rows that do not have "delivered" in the order_status column
non_delivered = orders_df[orders_df['order_status'] != 'delivered']
# unfilled/null rows even with "delivered" status
delivered_with_nulls = orders_df[
    (orders_df['order_status'] == 'delivered') &
    (orders_df.isnull().any(axis=1))
]
# Dropping unfilled rows even with "delivered" status
orders_df = orders_df.drop(delivered_with_nulls.index)

# checking for null values in the products DataFrame
empty_product_name = products_df[products_df['product_category_name'].isnull()]
product_ids_to_remove = empty_product_name['product_id']

#defining removing product_id
def remove_product_ids(df):
    # removing the rows with product_id
    df = df[~df['product_id'].isin(product_ids_to_remove)]
    return df
remove_product_ids(products_df)

# finding the 2 outliers, 2 null values in product_weight_g, product_length_cm, product_height_cm and product_width_cm
empty = products_df[products_df['product_weight_g'].isnull()]

# Add a new product_id to the product_ids_to_remove series using concat
product_ids_to_remove = pd.concat([product_ids_to_remove, pd.Series(['09ff539a621711667c43eba6a3bd8466'])], ignore_index=True)
# removing the row with null values in product_weight_g, product_length_cm, product_height_cm and product_width_cm
remove_product_ids(products_df)
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

# Rename columns
orderpaymentmerge.rename(columns={
    'payment_value': 'total_payment',
    'is_voucher': 'voucher_used'
}, inplace=True)

# Merge with voucher counts to get the number of vouchers used per order
orderpaymentmerge = pd.merge(orderpaymentmerge, voucher_counts, on='order_id', how='left')

# Fill missing 'voucher_count' values with 0 and convert to integer
orderpaymentmerge['voucher_count'] = orderpaymentmerge['voucher_count'].fillna(0).astype(int)

# Display the final result
order_payments_df = orderpaymentmerge

#Merging data:

#Merge Customer_df with Orders_df
customer_orders_df = pd.merge(customers_df, orders_df, on='customer_id', how="inner")

#merge customer, orders df with payment df
customer_orders_payment_df = pd.merge(customer_orders_df, order_payments_df, on='order_id', how="inner")
#Ensure that geolocation zipcodes are in customer dataset
filtered_customer_orders_payment_df = customer_orders_payment_df[customer_orders_payment_df['customer_zip_code_prefix'].isin(geolocation_df['geolocation_zip_code_prefix'])]
#merge the geolocation-filtered customer + order + payment dataset with reviews
filtered_customer_orders_payment_reviews_df = pd.merge(filtered_customer_orders_payment_df, order_reviews_df, on='order_id', how="inner")

# Before we merge the rest, merge Product cateogry name translated with Product_df first
products_df = products_df.merge(
    product_category_name_translation_df,
    on='product_category_name',
    how='left'
)
# Replace the original column with the English version
products_df['product_category_name'] = products_df['product_category_name_english']
# Drop the now redundant English translation column
products_df.drop(columns=['product_category_name_english'], inplace=True)
products_df = products_df[['product_id', 'product_category_name']]
#merge order items with product id-get product category for order items
products_order_items_df = order_items_df.merge(products_df, on='product_id', how='left')
# I want to merge the orders together, though they include different products, so i merged them together.
products_order_items_df_grouped_Version1 = products_order_items_df.groupby('order_id', as_index=False).agg({
    'product_category_name': lambda x: ', '.join(sorted(set(x.dropna()))),
    'price': 'sum',
    'freight_value': 'sum',
    'quantity': 'sum'
})

#merged dataset
merged_dataset = filtered_customer_orders_payment_reviews_df.merge(products_order_items_df_grouped_Version1, on='order_id', how='inner')
