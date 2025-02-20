# 🏥 Breast Cancer Classification Using Machine Learning

## 📌 Overview
Breast cancer remains a significant global health challenge, and early, accurate diagnosis is crucial for improving patient outcomes. This project leverages **machine learning techniques** to classify breast tumors as **malignant or benign** based on patient cell feature data. The primary goal is to identify **key features** that influence malignancy predictions and improve diagnostic accuracy.

## 🎯 Objective
- Develop a **machine learning model** that can accurately classify breast tumors.
- Use **Principal Component Analysis (PCA)** to reduce dimensionality and extract important features.
- Compare different classification algorithms including **Logistic Regression, Random Forest, SVM, and Gradient Boosting**.
- Visualize data distributions and model performance through plots and confusion matrices.

## 📊 Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features:** 30 numerical attributes describing characteristics of cell nuclei present in breast mass
- **Target Variable:** `diagnosis` (M = Malignant, B = Benign)

## 🚀 Methodology
1. **Data Preprocessing**
   - Handle missing values (if any)
   - Scale features using **StandardScaler**
   - Convert categorical labels to binary (M = 1, B = 0)
   
2. **Exploratory Data Analysis (EDA)**
   - Feature distribution analysis
   - Boxplots and scatter plots for key features
   - Correlation heatmap to identify influential attributes

3. **Dimensionality Reduction**
   - **PCA (Principal Component Analysis)** to retain 95% of variance
   - Visualization of first two principal components

4. **Model Training & Evaluation**
   - Algorithms tested:
     - **Logistic Regression**
     - **Random Forest Classifier**
     - **Support Vector Machine (SVM)**
     - **Gradient Boosting (XGBoost)**
   - Performance metrics:
     - **Accuracy, Precision, Recall, F1-score**
     - **Confusion Matrix Visualization**

## 📈 Results
| Model | Accuracy |
|--------|----------|
| Logistic Regression | 98% |
| Random Forest | 96% |
| SVM | 98% |
| Gradient Boosting | 96% |

✅ PCA improved model performance by reducing noise and computational complexity.
✅ Logistic Regression and SVM yielded the best accuracy (98%).

## 📂 Project Structure
```
Capstone/
│── data/                  # Dataset (if public) or link to Kaggle
│── notebooks/             # Jupyter notebooks for EDA & ML models
│── scripts/               # Python scripts for training & testing models
│── results/               # Plots, confusion matrices, PCA graphs
│── README.md              # Project documentation
│── requirements.txt       # Dependencies (pandas, scikit-learn, matplotlib, etc.)
```

## 🔧 Installation & Setup
1. **Clone this repository**
   ```bash
   git clone https://github.com/YannKEITA/Capstone.git
   cd Capstone
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate  # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 📌 Future Improvements
- Implement **Deep Learning** (Neural Networks) for improved classification.
- Explore additional **feature selection techniques** to optimize model efficiency.
- Enhance interpretability with **SHAP** and **LIME** explanations.

## 📫 Connect with Me
- **LinkedIn:** [Yann Keita](https://www.linkedin.com/in/yann-keita-76160a339/)
- **GitHub:** [YannKEITA](https://github.com/YannKEITA)
- **Email:** yatkeit@yahoo.fr

🚀 *This project is a step towards leveraging data science to improve cancer diagnostics and healthcare solutions!*
