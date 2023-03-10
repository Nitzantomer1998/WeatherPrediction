import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

weather_data = pd.read_csv(os.path.join("../../../data", "weather.csv"))

weather_features = weather_data.drop(["location", "date", "weather"], axis=1)

weather_features = weather_features.fillna(weather_features.mean())

scaler = StandardScaler()
weather_features = scaler.fit_transform(weather_features)

weather_labels = weather_data["weather"]

train_features, test_features, train_labels, test_labels = train_test_split(weather_features, weather_labels,
                                                                            test_size=0.2)

uniform_knn = KNeighborsClassifier(n_neighbors=10)

scores = cross_val_score(uniform_knn, weather_features, weather_labels, cv=5)

uniform_knn.fit(train_features, train_labels)

predicted_labels = uniform_knn.predict(test_features)

test_accuracy = round(accuracy_score(test_labels, predicted_labels), 4) * 100

print(f'Test accuracy = {test_accuracy}')
