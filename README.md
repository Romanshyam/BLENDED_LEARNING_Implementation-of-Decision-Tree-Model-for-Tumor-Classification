# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as pandas, sklearn modules, seaborn, and matplotlib.
2. Load the tumor dataset into a DataFrame using pandas.
3. Explore the dataset to understand its structure and features.
4. Preprocess the data by selecting relevant features and defining the target variable.
5. Split the dataset into training and testing sets using `train_test_split`.
6. Initialize the Decision Tree Classifier and train it using the training data.
7. Make predictions on the test set and evaluate the model using accuracy score, classification report, and confusion matrix.
8. Visualize the confusion matrix using a heatmap to interpret the model performance.


## Program:
```
Program to  implement a Decision Tree model for tumor classification.
Developed by: SHYAM KUMAR E 
RegisterNumber: 212223230207

# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset from the provided URL
file = "tumor.csv"
data = pd.read_csv(file)

# Step 2: Explore the dataset
# Display the first few rows and column names to verify the structure
print(data.head())
print(data.columns)

# Step 3: Select features and target variable
# Drop 'id' and other non-feature columns, using 'diagnosis' as the target
X = data.drop(columns=['Class'])  # Remove any irrelevant columns like 'id'
y = data['Class']  # The target column indicating benign or malignant diagnosis

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize and train the Decision Tree model
# Create a Decision Tree Classifier and fit it on the training data
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
# Predict on the test set and evaluate the results
y_pred = model.predict(X_test)

# Print the accuracy and classification metrics for the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Confusion Matrix
# Generate a heatmap of the confusion matrix for better visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
#### PREVIEW OF THE DATASET:
![image](https://github.com/user-attachments/assets/b04c91a8-f4a2-4784-a290-542939d48441)

#### EVALUATION METRICS:
![image](https://github.com/user-attachments/assets/e73c1fca-72c8-4d13-bc6a-641c87c53082)

#### CONFUSION MATRIX
![image](https://github.com/user-attachments/assets/6c75ea7e-1dd9-4af9-a805-96e358f6b403)


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
