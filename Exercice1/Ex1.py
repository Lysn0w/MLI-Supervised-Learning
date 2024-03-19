import numpy as np
import pandas as pd

np.random.seed(42)  # Assurer la reproductibilité

# Générer les données de base
n_points = 300
base_data = np.random.normal(loc=0, scale=1, size=(n_points, 6))

# Appliquer des transformations pour répondre aux exigences
# Ajuster les moyennes et les écarts-types
means = [2.5, 10, 20, -5, 0, 15]  # Une moyenne proche de 2.5, les autres distinctes
std_devs = [1, 2, 5, 10, 0.5, 3]  # Chaque écart type est différent

for i in range(base_data.shape[1]):
    base_data[:, i] = base_data[:, i] * std_devs[i] + means[i]

# Assurer les exigences de type de données
base_data[:, 0] = base_data[:, 0].round()  # Colonne avec des entiers
# Les autres colonnes sont déjà des flottants grâce aux transformations

# Ajouter des corrélations
# Colonne positivement corrélée (utiliser base_data[:, 1] avec du bruit)
base_data = np.hstack((base_data, (base_data[:, 1] * 1.2 + np.random.normal(0, 1, n_points)).reshape(-1, 1)))

# Colonne négativement corrélée (inverser base_data[:, 2] avec du bruit)
base_data = np.hstack((base_data, (-base_data[:, 2] * 0.8 + np.random.normal(0, 2, n_points)).reshape(-1, 1)))

# Colonne avec une corrélation proche de 0 (générer de nouvelles données indépendantes)
base_data = np.hstack((base_data, np.random.normal(50, 5, n_points).reshape(-1, 1)))

# Sauvegarder en CSV
df = pd.DataFrame(base_data, columns=[f"Colonne{i}" for i in range(1, base_data.shape[1] + 1)])
df.to_csv("artificial_dataset.csv", index=False)
