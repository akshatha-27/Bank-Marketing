# 🏦 Bank Marketing Prediction

## 📌 Project Overview

This project uses machine learning classification algorithms to predict whether a customer will subscribe to a term deposit based on demographic information and previous marketing campaign interactions.

Multiple classification models were implemented and evaluated using the Bank Marketing dataset. The selected Random Forest classifier was deployed as an interactive web application using Streamlit.

## 🎯 Objectives

- Analyze the Bank Marketing dataset.
- Perform data preprocessing and feature encoding.
- Implement and compare multiple machine learning classification models.
- Evaluate model performance using classification metrics.
- Deploy the selected model through a user-friendly web application.

## 📊 Dataset

The project uses the **Bank Marketing dataset**, which contains customer information and details about direct marketing campaigns conducted by a Portuguese banking institution.

**Target variable:** `y`

- `yes` – The customer subscribed to a term deposit.
- `no` – The customer did not subscribe to a term deposit.

The dataset includes demographic, financial, contact, and campaign-related attributes.

Dataset source: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

## ⚙️ Project Workflow

1. **Data Preprocessing:** Checked missing values, handled categorical values, encoded categorical features, and prepared the data for model training.
2. **Data Splitting:** Divided the dataset into training and testing sets.
3. **Model Implementation:** Trained multiple classification algorithms, including Gaussian Naive Bayes, Logistic Regression, Decision Tree, Random Forest, AdaBoost, and neural network-based classification.
4. **Model Evaluation:** Compared model performance using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
5. **Model Saving:** Saved the trained Random Forest classifier and fitted preprocessor using Joblib.
6. **Deployment:** Developed a Streamlit application and deployed it using Streamlit Community Cloud.

## 🤖 Machine Learning Models

The project explores multiple classification approaches:

- Gaussian Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Multilayer Perceptron (MLP)

The models were evaluated using the same classification evaluation framework to compare their predictive performance. The Random Forest classifier was selected for deployment.

## 🚀 Live Application

**Try the deployed application here:**

🔗 [Bank Marketing Prediction – Live Demo](https://bank-marketing-akshatha-nishmitha.streamlit.app/)

Users can enter customer details and obtain a prediction indicating whether the customer is likely to subscribe to a term deposit.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud
- **Development Environment:** Jupyter Notebook

## 📁 Repository Structure

```text
Bank-Marketing/
├── app.py
├── random_forest_model.pkl
├── preprocessor.pkl
├── requirements.txt
└── README.md
```

- `app.py` – Streamlit web application.
- `random_forest_model.pkl` – Saved Random Forest classifier.
- `preprocessor.pkl` – Saved fitted preprocessing pipeline.
- `requirements.txt` – Required Python libraries.
- `README.md` – Project documentation.

## 💻 Run the Project Locally

**1. Clone the repository**

```bash
git clone https://github.com/akshatha-27/Bank-Marketing.git
cd Bank-Marketing
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your default browser.

## 🌐 Deployment

The application was deployed using Streamlit Community Cloud.

1. Uploaded the application files to GitHub.
2. Connected the GitHub repository to Streamlit Community Cloud.
3. Selected `app.py` as the main application file.
4. Installed the required dependencies using `requirements.txt`.
5. Deployed and tested the live application.

## 📈 Evaluation Metrics

The classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics help assess overall classification performance and the model's ability to identify customers likely to subscribe.

## 🔮 Future Enhancements

- Improve prediction performance through hyperparameter tuning.
- Explore ensemble classification techniques.
- Enhance the application's interface and visualization features.
- Provide additional insights into factors influencing customer subscription.

## 👩‍💻 Author

**Akshatha Nayak**

MCA Student | Machine Learning | Python | Data Analytics

---

⭐ If you find this project useful, consider giving the repository a star.
