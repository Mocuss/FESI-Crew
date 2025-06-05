import joblib
import pandas as pd
import yaml
from dataprep.datapreprocessing import preprocess_data  # Importing the preprocess_data function from datapreprocessing.py

# Load the config.yaml file
def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to load the saved model
def load_model(model_path):
    # Load the model using joblib
    return joblib.load(model_path)

# Function to make predictions
def predict_new_data(config):
    # Step 1: Preprocess the new data
    print("Preprocessing the new data...")
    preprocessed_data = preprocess_data(
        config['data_sources']['customers'],
        config['data_sources']['geolocation'],
        config['data_sources']['order_items'],
        config['data_sources']['order_payments'],
        config['data_sources']['order_reviews'],
        config['data_sources']['orders'],
        config['data_sources']['products'],
        config['data_sources']['product_category_name_translation']
    )

    # Step 2: Load the saved model
    model = load_model(config['model']['path'])
    print("Model loaded successfully.")

    # Step 3: Make predictions
    predictions = model.predict(preprocessed_data)

    # Step 4: Output the predictions (for now we will print it)
    return predictions

# Main function to run the pipeline
if __name__ == '__main__':
    # Load the configuration
    config = load_config('config.yaml')
    
    # Step 5: Predict on new data using the pipeline
    predictions = predict_new_data(config)
    
    # Print out the predictions (You can adjust this to save to a file or database)
    print("Predictions:")
    print(predictions)
