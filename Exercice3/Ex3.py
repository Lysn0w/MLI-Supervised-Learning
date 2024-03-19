import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

# Charger les données
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialisation d'un dictionnaire pour stocker les scores de précision
accuracy_scores = {}

# Fonction pour entraîner et évaluer un modèle
def train_evaluate_model(model, name):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy
    print(f'Précision de {name}: {accuracy:.4f}')

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
train_evaluate_model(lr, 'Régression Logistique')

# Support Vector Classifier
svc = SVC()
train_evaluate_model(svc, 'SVC')

# K-Nearest Neighbors
knn = KNeighborsClassifier()
train_evaluate_model(knn, 'KNN')

# Multi-layer Perceptron classifier
mlp = MLPClassifier(max_iter=1000)
train_evaluate_model(mlp, 'MLP')

# AdaBoost
ada = AdaBoostClassifier()
train_evaluate_model(ada, 'AdaBoost')

# Afficher le modèle avec la meilleure précision
best_model = max(accuracy_scores, key=accuracy_scores.get)
print(f'Le meilleur modèle est {best_model} avec une précision de {accuracy_scores[best_model]:.4f}')