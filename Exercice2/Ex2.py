import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Charger le jeu de données
df = pd.read_csv('dataset.csv')


# Fonction pour calculer la dissimilarité catégorielle
def categorical_dissimilarity(val1, val2):
    return 0 if val1 == val2 else 1


# Fonction pour normaliser les données numériques manuellement
def normalize_data(df, num_indices):
    for i in num_indices:
        min_val = df.iloc[:, i].min()
        max_val = df.iloc[:, i].max()
        df.iloc[:, i] = (df.iloc[:, i] - min_val) / (max_val - min_val)
    return df


# Identifier les indices des caractéristiques numériques et catégorielles
num_indices = [i for i, dtype in enumerate(df.dtypes) if dtype in ['int64', 'float64']]
cat_indices = [i for i, dtype in enumerate(df.dtypes) if dtype == 'object']

# Normaliser les caractéristiques numériques manuellement
df = normalize_data(df, num_indices)


# Fonction pour calculer la dissimilarité entre deux échantillons
def custom_dissimilarity(sample1, sample2, num_indices, cat_indices):
    # Calculer la dissimilarité numérique
    num_dissimilarity = euclidean(sample1[num_indices], sample2[num_indices])

    # Calculer la dissimilarité catégorielle
    cat_dissimilarity = sum(categorical_dissimilarity(sample1[i], sample2[i]) for i in cat_indices)

    # Combinaison des dissimilarités
    total_dissimilarity = num_dissimilarity + cat_dissimilarity
    return total_dissimilarity


# Calcul de la matrice de dissimilarité
n_samples = len(df)
dissimilarity_matrix = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    for j in range(i + 1, n_samples):  # Optimisation : calculer seulement la moitié supérieure
        diss = custom_dissimilarity(df.iloc[i], df.iloc[j], num_indices, cat_indices)
        dissimilarity_matrix[i, j] = diss
        dissimilarity_matrix[j, i] = diss  # La matrice est symétrique

# Calculer la moyenne et l'écart type de la distribution de dissimilarité
mean_dissimilarity = np.mean(dissimilarity_matrix)
std_dissimilarity = np.std(dissimilarity_matrix)

# Sauvegarder la matrice de dissimilarité
np.save('dissimilarity_matrix.npy', dissimilarity_matrix)

print(f"Moyenne de la dissimilarité : {mean_dissimilarity}")
print(f"Écart type de la dissimilarité : {std_dissimilarity}")
