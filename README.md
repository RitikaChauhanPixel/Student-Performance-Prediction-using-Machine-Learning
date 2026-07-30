# 🎓 Student Performance Prediction using Machine Learning

An end-to-end Machine Learning project that predicts a student's exam score based on academic, personal, and lifestyle factors. The project covers the complete Machine Learning workflow, from data preprocessing and model training to deployment using an interactive Streamlit web application.

---

## 📖 Description

This project demonstrates the complete Machine Learning pipeline using a real-world student performance dataset. A Linear Regression model is trained to predict students' exam scores based on multiple input features such as study hours, attendance, previous scores, sleep hours, tutoring sessions, and more.

The trained model is integrated into a Streamlit web application where users can enter student details, generate predictions, explore analytics, and download prediction reports.

---

## ✨ Key Features

- 🎯 Predict student exam scores using Machine Learning
- 🌐 Interactive Streamlit web application
- 🧹 Data cleaning and preprocessing
- 🔄 Missing value handling
- 🔤 Label Encoding for categorical features
- 📏 Feature Scaling using StandardScaler
- 📈 Linear Regression model for prediction
- 📊 Model evaluation using MAE, MSE, RMSE, and R² Score
- 📉 Feature importance analysis
- 📌 Actual vs Predicted visualization
- 📍 Residual error analysis
- 💡 Personalized student performance insights
- 📜 Prediction history tracking
- 📥 Downloadable prediction reports (CSV)
- 📱 Responsive multi-page dashboard

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn (Linear Regression)

### Data Processing
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn
- Plotly

### Web Framework
- Streamlit

### Model Serialization
- Joblib

### Report Generation
- OpenPyXL
- ReportLab

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/RitikaChauhanPixel/Student-Performance-Prediction-using-Machine-Learning.git
```

### 2. Navigate to the Project Folder

```bash
cd Student-Performance-Prediction-using-Machine-Learning
```

### 3. Install the Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run App.py
```

### 5. Open in Browser

The Streamlit application will automatically open in your default web browser.

Enter the student's details and click the **Predict** button to generate the predicted exam score and view the analytics.

---

## 📂 Project Structure

```
Student Performance Prediction using Machine Learning/
│
├── Dataset/
├── Models/
├── Reports/
├── Images/
│
├── ReadDataSet.py
├── FinalModel.py
├── PredictStudent.py
├── App.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ✅ Project Status

**Completed**

---

## 👩‍💻 Author

**Ritika Chauhan**

Computer Engineering Student • Machine Learning Enthusiast

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.