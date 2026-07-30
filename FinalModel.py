import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib #For Model Saving

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)


# Dataset Loading
data = pd.read_csv("Dataset/CleanedDataset.csv")
df = pd.DataFrame(data)
print(df)
print("Dataset Loaded Successfully")




# Independent Variables - X
# Dependent Variable - Y
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestRegressor(random_state=42,n_estimators=100)
rf.fit(X_train, y_train)


# Prediction from all 3 of them
lr_pred = lr.predict(X_test_scaled)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)


# Evaluation
def evaluate(name, y_true, y_pred):
    print(name)
    print("MAE      :", mean_absolute_error(y_true, y_pred))
    print("MSE      :", mean_squared_error(y_true, y_pred))
    print("RMSE     :", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2 Score :", r2_score(y_true, y_pred))

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Decision Tree", y_test, dt_pred)
evaluate("Random Forest", y_test, rf_pred)

# Finding Importance of each feature
coef_df = pd.DataFrame({
    "Feature": X.columns,  #Ex - Attendance - 2.278393120319921
    "Coefficient": lr.coef_
})
coef_df = coef_df.sort_values(by="Coefficient",ascending=False)
print("Feature Importance")
print(coef_df)

coef_df.to_csv("Reports/Feature_Importance.csv",index=False)

# Actual Vs Predicted Graph
plt.figure(figsize=(7,7))
plt.scatter(y_test,lr_pred,alpha=0.7)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red",linewidth=2)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Score")
plt.grid(True)
plt.savefig(
    "Images/Actual_vs_Predicted.png",
    dpi=300
)
plt.show()

# Residual = Actual − Predicted
residuals = y_test - lr_pred
plt.figure(figsize=(7,5))
plt.scatter(lr_pred,residuals,alpha=0.7)
plt.axhline(y=0,color="red",linestyle="--")
plt.xlabel("Predicted Exam Score")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.savefig(
    "Images/Residual_Plot.png",
    dpi=300
)
plt.show()

# Actual Vs Predicted Data
prediction_report = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": np.round(lr_pred,2)
})
prediction_report["Difference"] = np.round(      #New col for difference
    prediction_report["Actual Marks"]
    - prediction_report["Predicted Marks"],2
)
prediction_report.to_csv("Reports\Prediction_Report.csv",   index=False
)

print("Prediction Report")
print(prediction_report.head(10))

# Model Saving For after usage
joblib.dump(lr,"Models\StudentMarksPredictor.pkl")
joblib.dump(scaler,"Models\ModelsScaler.pkl")
print("Model Saved Successfully")
print("Scaler Saved Successfully")

# print("Sample Prediction")
# sample_student = X.iloc[[0]]
# sample_scaled = scaler.transform(sample_student)
# prediction = lr.predict(sample_scaled)
# print("Actual Marks    :", y.iloc[0])
# print("Predicted Marks :", round(prediction[0],2))