# Import necessary libraries
import os  # For working with file paths

import pandas as pd  # For data loading and manipulation
from sklearn.metrics import accuracy_score  # For computing classification accuracy
from sklearn.model_selection import cross_val_score, train_test_split  # For cross-validation and data splitting
from sklearn.naive_bayes import GaussianNB  # For Gaussian Naive Bayes classification
from sklearn.preprocessing import StandardScaler  # For feature scaling

# Load the data from the CSV file
weather_data = pd.read_csv(os.path.join("../../../../data", "weather.csv"))

# Preprocess the data
# Drop columns that are not needed
weather_features = weather_data.drop(["location", "date", "weather"], axis=1)

# Fill in missing values with the column means
weather_features = weather_features.fillna(weather_features.mean())

# Scale numerical features
scaler = StandardScaler()
weather_features = scaler.fit_transform(weather_features)

# Define the output column of the data
weather_labels = weather_data["weather"]

# Create the model
gaussian_naive_bayes = GaussianNB()

# Use cross-validation to estimate model performance
# cv=5 specifies 5-fold cross-validation
scores = cross_val_score(gaussian_naive_bayes, weather_features, weather_labels, cv=5)

# Split the data into training and testing sets
# test_size=0.2 specifies a 20% testing set and 80% training set
train_features, test_features, train_labels, test_labels = train_test_split(weather_features, weather_labels,
                                                                            test_size=0.2)

# Train the model on the training set
gaussian_naive_bayes.fit(train_features, train_labels)

# Make predictions on the testing set
predicted_labels = gaussian_naive_bayes.predict(test_features)

# Compute the accuracy score for the model on the testing set
test_accuracy = round(accuracy_score(test_labels, predicted_labels), 4) * 100

# Print the accuracy score for the model on the testing set
print(f'Test accuracy = {test_accuracy}')
