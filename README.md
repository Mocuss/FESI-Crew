# FESI-Crew
EGT309 Project <br>
<hr>
<h1>1. Members: </h1> <br>
1. Samuel Kosasih <br>
2. Lee Yit Rhong Mattias, 231988D@mymail.nyp.edu.sg<br>
3. Ryan Yong<br>
4. Marcus Teo Ming Xuan, 230208B@mymail.nyp.edu.sg<br>
<hr>
<h1>2. Folder Structure:</h1>
<ul>
<li>src:<ol>
<li>dataprep: folder containing data preprocessing
<ul>
<li>datapreprocessing.py: contains the code to clean, merge and feature engineer the datasets to be machine learning ready. </li></ul></li>
<li>model: folder containing model training and testing, along with predictions.
<ul>
<li>prediction.py: Where our predictions are tested. </li>
<li>train_test.ipynb: Where our machine learning model is trained and tested. </li>
</ul>
</li>


</ol>
</li>
<li>config.yaml: Configuration file for dataset paths and model path </li>
<li>EDA Dashboard: View insights on the data </li>
<li>EDA.ipynb: Exploration Data Analysis, where we cleaned, merged, explored and did feature engineering to our data</li>
<li>EDA.pdf: Exploration Data Analysis, PDF version</li>
<li>requirements.txt: tools and libraries needed to be installed to run the project </li>
<li>run.sh: Shell script </li>
<li>Dockerfile: </li>
</ul>
<hr>
<h1>3.Programming Language </h1>
The following are libraries needed to run our project:
The following are needed to run our project:<br>
Language: Python 3.8 or above<br>
Libraries:<br>
<ul>
<li>`pandas==1.3.4`</li>
<li>`numpy==1.21.2`</li>
<li>`xgboost==1.5.1`</li>
<li>`scikit-learn==0.24.2``</li>
<li>`scipy==1.7.1`</li>
<li>`pyyaml==5.4.1`</li>
<li>`joblib==1.0.1`</li>
<li>`matplotlib==3.4.3</li>
</ul>


<hr>

<h1>4. Key EDA findings </h1>
<h2>Data Cleaning</h2>
We first checked for null and duplicated rows within our datasets. Then, based on olist_datadict.xlsx metadata, we checked and corrected the datasets such that it not only correlates with olist_datadict.xlsx, but also makes merging and future data alterations easy and clear.
<h2>Feature Engineering</h2>
Our feature engineering boils down to five core customer signals:
Recency
 How recently someone shopped: we calculate the number of days between today (our reference date) and each customer’s most recent purchase. Shorter gaps → more engaged.


Frequency
 How often they buy: we count the distinct orders each customer placed in the six-month window. More orders → higher loyalty.


Monetary
 How much they spend: we sum up their order values (total spend) and divide by order count (average basket size) over that same period. Big spenders and consistent basket sizes are prime targets.


Behaviour
 • Delivery – for each order we flag on-time vs late and measure shipping delay in days, then average per customer to get an on-time ratio and mean/variance of delay.
 • Reviews – we join in all review scores, then compute each customer’s average rating and review count. Positive feedback and active reviewers tend to return.


Target
 Our binary label: did the customer make more than one purchase? Customers with ≥2 orders are marked “returning,” and that flag is what our model learns to predict.


Together, these five signals capture when, how often, how much, how well, and whether again—the exact dimensions needed to spot your next repeat buyer.
<h2>Insights found</h2>
The following are insights found from our Power BI analysis, refer to our dashboard and slides for additional information: <br>
1. Returning customers(0.11 in Q1) are more likely to pay with vouchers as compared to non returning customers (0.05). <br>
2. Returning customers (Approx. 1.3 in Q1) tend to order more items as compared to non returning customers (Approx. 1.2 in Q1).
3.Returning customers are slightly more satisfied than non returning customers. 63% of returning voted for 5 star reviews, while only 57% on non-returning

<hr>

✅ Prerequisites

Ensure the following are installed on your system:

    Docker

    git

    (Optional) dos2unix (if developing on Windows)

📦 1. Clone the Repository

git clone https://github.com/Mocuss/FESI-Crew.git
cd FESI-Crew

🛠️ 2. Fix run.sh Line Endings (Once Only)

If you cloned the repo on Windows, convert the script to Unix format:

dos2unix run.sh
chmod +x run.sh

    This prevents command not found: \r errors when running the shell script.

🐳 3. Build the Docker Image

docker build -t fesi-predictor .

This will:

    Install dependencies from requirements.txt

    Copy source code and config files

    Prepare your prediction system

📝 4. Edit config.yaml

Update the file paths in config.yaml to your local CSV data files and model (if needed):

data_sources:
  customers: "/app/data/olist_customers_dataset.csv"
  geolocation: "/app/data/olist_geolocation_dataset.csv"
  ...
model:
  path: "/app/saved_model/best_model.pkl"

You can either:

    Edit directly inside your local config.yaml, or

    Enter a running container and modify from inside:

docker run -it fesi-predictor /bin/bash
nano config.yaml  # or vi

🚀 5. Run the System

Once config.yaml is configured, run the container:

docker run -it fesi-predictor

This will:

    Run run.sh

    Perform preprocessing

    Load the model

    Run prediction

    Print out output to terminal or save it

### Step 2: Run entire pipeline (requires yq and config.yaml)
bash run.sh<br>


### Alternatively run manually:
python src/dataprep/datapreprocessing.py --args...<br>
python src/model/prediction.py --args… <br>


<hr>


<h1>6. Pipeline Flow </h1>
Data is linked to the config file <br>
 Data is cleaned, merged and feature engineered through data preprocessing <br>
Model trained and predictions made <br>
<hr>


<h1>7. Choice of Model</h1>
For our model, we decided to use XGBoost and Random Forest. As our scenario is a binary class classification problem (Returning Customer or Non-returning customer), regression models like linear regression would not work as well. <br>
Random Forest is well suited for large datasets (90k data), and is more resistant to overfitting. <br>
XGBoost is not only great for non-linear relationships, but also provides feature importance as well, allowing us to evaluate which feature provided the most value. This feature creates a transparent model.  XGBoost is also scalable and handles large amount of data well.<br>
<hr>

<h1>8. Model Evaluation </h1>
We decided to focus on overall accuracy, weighted precision, weighted recall and weighted f1 score as there is a huge imbalance between non returning and returning customers. The use of the weighted matrix allows us to account for the imbalance and hence is a better matrix to use. <br>
<h2>For Random Forest: </h2><br>
Overall Accuracy: 0.94<br>
Weighted Precision: 0.95 <br>
Weighted Recall: 0.94<br>
Weighted F1 Score: 0.49<br>

<h2>For XGBoost: </h2><br>
Overall Accuracy: 0.94 <br>
Weighted Precision: 0.91<br>
Weighted Recall: 0.94<br>
Weighted F1 Score: 0.92<br>


From this, we can see that our model can reliably differentiate between returning and non returning customers as the accuracy shown above is >90%. <br>


<hr>
<h1>9.Additional Considerations </h1>
<h2>Saved Model</h2> <br>
Our model is too big to be placed in GitHub, please do use this link to download the model instead. <br>
https://drive.google.com/drive/folders/1j_Trz1raShsTIUm8q8tP7eEF_3-YtlHa?usp=sharing <br>


