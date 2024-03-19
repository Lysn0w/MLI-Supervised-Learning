import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

# Charger les ensembles de données
X_train = np.load('X_train-2.npy')
X_test = np.load('X_test-2.npy')
y_train = np.load('y_train-2.npy').ravel()  # Conversion en 1D
y_test = np.load('y_test-2.npy').ravel()

# Optimisation du modèle Lasso avec un pipeline
lasso_pipeline = make_pipeline(StandardScaler(), Lasso(max_iter=10000))
lasso_params = {
    'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_search_lasso = GridSearchCV(lasso_pipeline, lasso_params, cv=5, scoring='r2')
grid_search_lasso.fit(X_train, y_train)

# Optimisation du MLPRegressor avec un pipeline
mlp_pipeline = make_pipeline(StandardScaler(), MLPRegressor(max_iter=10000))
mlp_params = {
    'mlpregressor__hidden_layer_sizes': [(100,), (100, 100), (50, 50, 50)],
    'mlpregressor__alpha': [0.0001, 0.001, 0.01]
}
grid_search_mlp = GridSearchCV(mlp_pipeline, mlp_params, cv=5, scoring='r2')
grid_search_mlp.fit(X_train, y_train)

# Affichage des meilleurs paramètres et score R² sur l'ensemble de test pour Lasso
print(f"Lasso - Meilleurs paramètres: {grid_search_lasso.best_params_}")
y_pred_lasso = grid_search_lasso.predict(X_test)
print(f"Lasso - Score R² sur l'ensemble de test: {r2_score(y_test, y_pred_lasso):.4f}")

# Affichage pour MLPRegressor
print(f"MLPRegressor - Meilleurs paramètres: {grid_search_mlp.best_params_}")
y_pred_mlp = grid_search_mlp.predict(X_test)
print(f"MLPRegressor - Score R² sur l'ensemble de test: {r2_score(y_test, y_pred_mlp):.4f}")

# Initialiser un dictionnaire pour stocker les scores R²
scores_r2 = {
    'Lasso': r2_score(y_test, y_pred_lasso),
    'MLPRegressor': r2_score(y_test, y_pred_mlp)
}

# Trouver le modèle avec le meilleur score R²
meilleur_modele = max(scores_r2, key=scores_r2.get)
meilleur_score = scores_r2[meilleur_modele]

# Afficher le modèle le plus efficace et son score R²
print(f"La méthode la plus efficace est {meilleur_modele} avec un score R² de {meilleur_score:.4f} sur l'ensemble de test.")

