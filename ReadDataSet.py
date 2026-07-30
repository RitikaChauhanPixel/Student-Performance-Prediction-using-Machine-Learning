import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder

# ==========================
# Step 1 : Load Dataset
# ==========================

data = pd.read_csv("Dataset/StudentPerformanceFactors.csv")
df = pd.DataFrame(data)

# ==========================
# Step 2 : Handle Missing Values
# ==========================

df["Teacher_Quality"] = df["Teacher_Quality"].fillna(
    df["Teacher_Quality"].mode()[0]
)

df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
    df["Parental_Education_Level"].mode()[0]
)

df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
    df["Distance_from_Home"].mode()[0]
)

print("Missing Values")
print(df.isnull().sum())

# ==========================
# Step 3 : Exploratory Data Analysis
# ==========================

plt.figure(figsize=(6,4))
plt.scatter(df["Hours_Studied"], df["Exam_Score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.grid(True)
plt.show()

# ==========================
# Step 4 : Label Encoding
# ==========================

categorical_cols = df.select_dtypes(include="object").columns

# Dictionary to store all LabelEncoders
label_encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    # Save encoder of this column
    label_encoders[col] = le

# ==========================
# Step 5 : Save Cleaned Dataset
# ==========================

df.to_csv(
    "Dataset/CleanedDataset.csv",
    index=False
)

# ==========================
# Step 6 : Save All Encoders
# ==========================

joblib.dump(
    label_encoders,
    "Models/LabelEncoders.pkl"
)

print("\nCleaned Dataset Saved Successfully.")
print("Label Encoders Saved Successfully.")

# ==========================
# Step 7 : Show Encoding Mapping
# ==========================

print("\nEncoding Mapping")

for col, encoder in label_encoders.items():

    print(f"\n{col}")

    for i, class_name in enumerate(encoder.classes_):

        print(f"{class_name} --> {i}")

print("\nFirst 5 Rows of Encoded Dataset")
print(df.head())