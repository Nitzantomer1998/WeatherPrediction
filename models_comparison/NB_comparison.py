# Imports necessary libraries
import os  # For working with file paths

import matplotlib.pyplot as plt  # For graph making
import numpy as np  # For mathematical
import pandas as pd  # For data loading and manipulation
from sklearn.model_selection import train_test_split  # For data splitting
from sklearn.naive_bayes import GaussianNB  # For NB classification

# Load and preprocess the data
weather_data = pd.read_csv(os.path.join("../data", "weather.csv"))
weather_features = weather_data.drop(["location", "date", "weather"], axis=1)
weather_features = weather_features.fillna(weather_features.mean())
weather_labels = weather_data["weather"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(weather_features, weather_labels, test_size=0.2, random_state=42)

# Define the two models with different priors
estimate_nb = GaussianNB()
prior_nb = GaussianNB(priors=[0.2, 0.2, 0.2, 0.2, 0.2])

# Compare the accuracy of the two models over 50 trials
num_trials = 50
estimate_nb_accuracy = []
prior_nb_accuracy = []

for i in range(num_trials):
    # Shuffle the training data
    idx = np.random.permutation(len(X_train))
    X_train_shuffled, y_train_shuffled = X_train.iloc[idx], y_train.iloc[idx]

    # Fit the models on the shuffled training data
    estimate_nb.fit(X_train_shuffled, y_train_shuffled)
    prior_nb.fit(X_train_shuffled, y_train_shuffled)

    # Compute the accuracy on the test data
    estimate_nb_accuracy.append(estimate_nb.score(X_test, y_test))
    prior_nb_accuracy.append(prior_nb.score(X_test, y_test))

# Plot the results
models = ['Estimate Prior', '1/K Prior']
x_pos = np.arange(len(models))
means = [np.mean(estimate_nb_accuracy) * 100, np.mean(prior_nb_accuracy) * 100]
stds = [np.std(estimate_nb_accuracy) * 100, np.std(prior_nb_accuracy) * 100]

# Create a bar plot with error bars for both models
fig, ax = plt.subplots()
ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10, width=0.5,
       color=['#1f77b4', '#ff7f0e'])

# Set the graph properties
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('Accuracy % Over 50 Trials')
ax.set_ylim([0, 100])
ax.set_title('NB gaussian Estimate VS 1/K Prior')
ax.yaxis.grid(True)

# Add the accuracy percentage on top of each bar
for i, v in enumerate(means):
    ax.text(i, v + 1, str(round(v, 2)), color='black', ha='center')

# Show the graph
plt.show()
