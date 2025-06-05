# FESI-Crew
EGT309 Project <br>
<hr>
<h1>1. Members: </h1> <br>
1. Samuel Kosasih <br>
2. Lee Yit Rhong Mattias<br>
3. Ryan Yong<br>
4. Marcus Teo Ming Xuan<br>
<hr>
<h1>2. Folder Structure:</h1>
<ul>
<li><ol>
src:

<li>dataprep: folder containing data preprocessing</li>
<li>        datapreprocessing.py: contains the code to clean, merge and feature engineer the datasets to be machine learning ready. </li>
<li>model: folder containing model training and testing, along with predictions.</li>
<li>        prediction.py: Where our predictions are tested. </li>
<li>        train_test.ipynb: Where our machine learning model is trained and tested. </li>

</ol>
</li>
<li>config.yaml: Configuration file for dataset paths and model path </li>
<li>EDA.ipynb: Exploration Data Analysis, where we cleaned, merged, explored and did feature engineering to our data</li>
<li>EDA.pdf: Exploration Data Analysis, PDF version</li>
<li>requirements.txt: tools and libraries needed to be installed to run the project </li>
<li>run.sh: Shell script </li>
<li>Dockerfile: </li>
</ul>
<hr>
<h1>3.Programming Language </h1>
The following are libraries needed to run our project:

<hr>

<h1>4. Key EDA findings </h1>
<h2>Data Cleaning</h2>
<h2>Feature Engineering</h2>
<h2>Insights found</h2>
The following are insights found from our Power BI analysis: <br>
<hr>

<h1>5.Instructions to run </h1>
<hr>

<h1>6. Pipeline Flow </h1>
<hr>

<h1>7. Choice of Model</h1>
For our model, we decided to use XGBoost and Random Forest. As our scenario is a binary class classification problem (Returning Customer or Non-returning customer), regresison models like linear gression would not work as well. <br>
Random Forest is well suited for large datasets (90k data), and is more resistent to overfitting. <br>
XGBoost is not only great for non-linear relationships, but also provides feature importance as well, allowing us to evaluate on which feature provided the most value.<br>
<hr>

<h1>8. Model Evaluation </h1>
<h2>For Random Forest: </h2><br>
Overall Accuracy: <br>
Weighted Average: <br>
Weighted Precision: <br>
Weighted Recall:<br>
Weighted F1 Score:<br>


<hr>
<h1>9.Additional Considerations </h1>
<hr>
<h1>Saved Model</h1> <br>
Our model is too big to be placed in GitHub, please do use this link to download the model instead. <br>
https://drive.google.com/drive/folders/1j_Trz1raShsTIUm8q8tP7eEF_3-YtlHa?usp=sharing <br>

