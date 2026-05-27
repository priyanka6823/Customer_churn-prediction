<h1 align="center"> AI-Powered Customer Retention Prediction System</h1>

<p align="center">
  Machine Learning based Telco Customer Churn Prediction Web Application
</p>

<hr>

<h2> Project Overview</h2>

<p>
This project predicts whether a telecom customer is likely to churn using Machine Learning algorithms.
The system analyzes customer information such as contract type, billing details, internet services,
tenure, and payment methods to identify customers who may discontinue services.
</p>

<p>
The final model is deployed using a Flask web application for real-time customer churn prediction.
</p>

<hr>

<h2> Features</h2>

<ul>
  <li>Customer Churn Prediction</li>
  <li>Machine Learning Pipeline</li>
  <li>Data Preprocessing & Feature Engineering</li>
  <li>Handling Missing Values & Outliers</li>
  <li>SMOTE Data Balancing</li>
  <li>Feature Scaling & Encoding</li>
  <li>Flask Web Application</li>
  <li>Real-Time Prediction</li>
</ul>

<hr>

<h2>🛠️ Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>Pandas</li>
  <li>NumPy</li>
  <li>Scikit-Learn</li>
  <li>XGBoost</li>
  <li>Flask</li>
  <li>HTML</li>
  <li>CSS</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
Telco-Customer-Churn/
│
├── app.py
├── main.py
├── all_models.py
├── feature_scaling.py
├── filter_method.py
├── median_imputation.py
├── var_tras.py
├── ca_to_num.py
├── model.pkl
├── scaler.pkl
├── templates/
│   └── index.html
├── static/
├── dataset/
└── README.md
</pre>

<hr>

<h2>⚙️ Machine Learning Workflow</h2>

<ol>
  <li>Data Collection</li>
  <li>Handling Missing Values</li>
  <li>Feature Engineering</li>
  <li>Variable Transformation</li>
  <li>Outlier Handling</li>
  <li>Categorical Encoding</li>
  <li>Feature Selection</li>
  <li>SMOTE Data Balancing</li>
  <li>Feature Scaling</li>
  <li>Model Training</li>
  <li>Model Evaluation</li>
  <li>Deployment using Flask</li>
</ol>

<hr>

<h2>  Models Used</h2>

<ul>
  <li>Logistic Regression</li>
  <li>K-Nearest Neighbors</li>
  <li>Naive Bayes</li>
  <li>Decision Tree</li>
  <li>Random Forest</li>
  <li>AdaBoost</li>
  <li>Gradient Boosting</li>
  <li>XGBoost</li>
  <li>Support Vector Machine (SVM)</li>
</ul>

<hr>

<h2> Final Selected Model</h2>

<p>
✔ Logistic Regression was selected as the final deployment model because of:
</p>

<ul>
  <li>High Accuracy</li>
  <li>Fast Prediction Speed</li>
  <li>Good ROC-AUC Score</li>
  <li>Better Interpretability</li>
</ul>

<hr>

<h2> Data Preprocessing Techniques</h2>

<ul>
  <li>Median Imputation</li>
  <li>Yeo-Johnson Transformation</li>
  <li>Quantile Transformation</li>
  <li>IQR Outlier Capping</li>
  <li>Label Encoding</li>
  <li>Ordinal Encoding</li>
  <li>One-Hot Encoding</li>
  <li>StandardScaler</li>
  <li>SMOTE Balancing</li>
</ul>

<hr>

<h2>🌐 Web Application</h2>

<p>
The Flask web application allows users to:
</p>

<ul>
  <li>Enter customer details</li>
  <li>Predict customer churn instantly</li>
  <li>View prediction results in real time</li>
</ul>

<p>
🔗 Live Demo:
<a href="https://customer-churn-prediction-idk0.onrender.com">
https://customer-churn-prediction-idk0.onrender.com
</a>
</p>

<hr>

<h2> Dataset</h2>

<p>
Dataset used:
Telco Customer Churn Dataset
</p>

<p>
Source:
<a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">
Kaggle Dataset
</a>
</p>

<hr>

<h2> How to Run the Project</h2>

<h3>1️ Clone Repository</h3>

<pre>
git clone https://github.com/your-username/telco-customer-churn.git
</pre>

<h3>2️ Install Requirements</h3>

<pre>
pip install -r requirements.txt
</pre>

<h3>3️ Run Flask Application</h3>

<pre>
python app.py
</pre>

<h3>4️ Open Browser</h3>

<pre>
http://127.0.0.1:5000/
</pre>

<hr>

<h2> Future Enhancements</h2>

<ul>
  <li>Deep Learning Models</li>
  <li>Interactive Dashboard</li>
  <li>Batch Prediction using CSV Upload</li>
  <li>Model Explainability using SHAP</li>
  <li>Real-Time Data Pipeline</li>
</ul>

<hr>

<h2> Author</h2>

<p>
<b>Priyanka</b><br>
Data Science Project
</p>

<hr>

<h2>📜 License</h2>

<p>
This project is created for educational and learning purposes.
</p>
