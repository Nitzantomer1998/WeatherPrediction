import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

weather_data = pd.read_csv(os.path.join("../data", "weather.csv"))
weather_features = weather_data.drop(["location", "date", "weather"], axis=1)
weather_features = weather_features.fillna(weather_features.mean())
weather_labels = weather_data["weather"]

X_train, X_test, y_train, y_test = train_test_split(weather_features, weather_labels, test_size=0.2, random_state=42)

num_trials = 50
k_values = [1, 5, 10, 15, 20, 25, 30]

accuracy_uniform_knn = []
accuracy_distance_knn = []

for k_value in k_values:
    uniform_accuracy_k = []
    distance_accuracy_k = []

    for i in range(num_trials):
        idx = np.random.permutation(len(X_train))
        X_train_shuffled, y_train_shuffled = X_train.iloc[idx], y_train.iloc[idx]

        uniform_knn = KNeighborsClassifier(n_neighbors=k_value)
        distance_knn = KNeighborsClassifier(n_neighbors=k_value, weights='distance')

        uniform_knn.fit(X_train_shuffled, y_train_shuffled)
        distance_knn.fit(X_train_shuffled, y_train_shuffled)

        uniform_accuracy_k.append(uniform_knn.score(X_test, y_test))
        distance_accuracy_k.append(distance_knn.score(X_test, y_test))

    accuracy_uniform_knn.append(np.mean(uniform_accuracy_k) * 100)
    accuracy_distance_knn.append(np.mean(distance_accuracy_k) * 100)

models = ['Uniform KNN', 'Distance KNN']
x_pos = np.arange(len(k_values))

means = [np.mean(accuracy_uniform_knn), np.mean(accuracy_distance_knn)]

fig, ax = plt.subplots()
ax.bar(x_pos - 0.2, means[0], yerr=0, align='center', alpha=0.5, ecolor='black', capsize=5, width=0.4, color='#1f77b4')
ax.bar(x_pos + 0.2, means[1], yerr=0, align='center', alpha=0.5, ecolor='black', capsize=5, width=0.4, color='#ff7f0e')

ax.set_xticks(x_pos)
ax.set_xticklabels(k_values)
ax.set_xlabel('K-Neighbors')
ax.set_ylabel('Accuracy % Over 50 Trials')
ax.set_ylim([0, 100])
ax.set_title('Uniform VS Distance KNN')
ax.yaxis.grid(True)
ax.legend(models, loc='upper right')

for i in range(len(k_values)):
    ax.text(i - 0.2, means[0] + 1, str(round(accuracy_uniform_knn[i], 1)), color='black', ha='center', fontsize=7)
    ax.text(i + 0.2, means[1] + 1, str(round(accuracy_distance_knn[i], 1)), color='black', ha='center', fontsize=7)

plt.show()
