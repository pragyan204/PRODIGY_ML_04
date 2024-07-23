import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from skimage.feature import hog
from skimage.color import rgb2gray

# Load dataset
food_path = r"C:\Users\PRAGYAN\Desktop\ProdigyProjects\food_path"
df = pd.read_csv('food_path')

# Function to process images and extract HOG features
def get_image_features(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (64, 64))
    gray_img = rgb2gray(img_resized)
    features, _ = hog(gray_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features

# Extract features and labels from dataset
feature_list = []
label_list = []
calorie_list = []

for index, row in df.iterrows():
    img_path = row['image_path']
    feature_list.append(get_image_features(img_path))
    label_list.append(row['food_item'])
    calorie_list.append(row['calories'])

# Convert lists to numpy arrays
X_data = np.array(feature_list)
y_labels = np.array(label_list)
y_calories = np.array(calorie_list)

# Encode the categorical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_labels)

# Split dataset into training and testing sets
X_train, X_test, y_train_labels, y_test_labels, y_train_calories, y_test_calories = train_test_split(
    X_data, encoded_labels, y_calories, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM for classification
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train_scaled, y_train_labels)

# Train Linear Regression model for calorie estimation
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train_calories)

# Predict on the test data
class_predictions = classifier.predict(X_test_scaled)
calorie_predictions = regressor.predict(X_test_scaled)

# Evaluate the models
print("Classification Accuracy:", accuracy_score(y_test_labels, class_predictions))
print("Classification Report:")
print(classification_report(y_test_labels, class_predictions))

# Evaluate regression model
mse = mean_squared_error(y_test_calories, calorie_predictions)
print(f"Mean Squared Error for Calorie Prediction: {mse}")

# Function to make predictions for a new image
def predict_food_and_calories(image_path):
    features = get_image_features(image_path)
    features_scaled = scaler.transform([features])
    predicted_label_encoded = classifier.predict(features_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
    predicted_calories = regressor.predict(features_scaled)
    return predicted_label[0], predicted_calories[0]

# Example of prediction
food_item, calorie_estimate = predict_food_and_calories('path_to_new_image.jpg')
print(f"Predicted Food Item: {food_item}")
print(f"Estimated Calories: {calorie_estimate}")
