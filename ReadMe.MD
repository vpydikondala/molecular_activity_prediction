# 🧬 Molecular Activity Prediction Using Machine Learning

This project demonstrates an end-to-end pipeline for predicting **molecular activity** (e.g. *NR-AhR receptor activation*) using molecular fingerprint data. It uses **Random Forest** and **XGBoost** classifiers, with **SMOTE** for handling class imbalance and **GridSearchCV** for hyperparameter tuning.

---

## 📁 Project Structure

molecular_activity_prediction/
│
├── data/
│ └── fingerprints.csv # Input dataset (place your file here)
│
├── src/
│ ├── data_preprocessing.py # Data loading and cleaning
│ ├── model_training.py # Model training, tuning, and evaluation
│ └── utils.py # Utilities like SMOTE
│
├── molecular_activity_prediction.ipynb # Jupyter notebook (end-to-end)
├── requirements.txt # Python dependencies
└── README.md # This file


## ✅ Features

- Load and preprocess fingerprint data
- Train/test split with stratification
- Handle imbalanced data using **SMOTE**
- Train models: **Random Forest** and **XGBoost**
- Perform **hyperparameter tuning** with GridSearchCV
- Evaluate models using classification metrics

---

## 🚀 Quickstart

### 1. Clone the repo (or download)


git clone https://github.com/yourusername/molecular_activity_prediction.git
cd molecular_activity_prediction
2. Setup Python environment (recommended)

python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Prepare the data
Place your fingerprints.csv file in the data/ folder.

📌 Ensure it contains the column NR-AhR as the target (0 or 1).

4. Run the notebook

jupyter notebook molecular_activity_prediction.ipynb
🧪 Dataset
The dataset consists of molecular fingerprints as numerical features and a binary target label (e.g. NR-AhR receptor activity).

Rows: Molecules

Columns: Fingerprint bits + target

Example: 2059 features + 1 target column

📊 Output Metrics
Accuracy

Precision, Recall, F1-score

Classification report per class

Best hyperparameters for each model

🛠 Dependencies
Install using the provided requirements.txt:


pandas
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
📌 Notes
If the dataset is highly imbalanced, performance on the minority class may suffer. This project uses SMOTE to address that.

You can replace NR-AhR with other target columns to predict different properties.

📁 Sample Output

Best RF parameters: {...}
Best RF CV accuracy: 0.95
Classification Report:
    Precision, Recall, F1 for both classes (0 and 1)
👨‍💻 Author
Venkanna Pydikondala
Email: venkanna.pydikondala@gmail.com

📅 Project Duration
Completed as part of self-learning + hands-on ML project using cheminformatics data.

📚 License
This project is open-source and freely usable for educational or research purposes.