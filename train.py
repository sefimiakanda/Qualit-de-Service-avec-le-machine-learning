import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Charger le dataset
print("Chargement des données...")

df = pd.read_excel('dataset_tp_ml.xlsx')

# 2. Vérifier l'absence de valeurs nulles
df = df.dropna()

# 3. Encodage des variables catégorielles (Texte en Nombres)

encodeurs = {} 
colonnes_texte = ['Opérateur', 'Quartier', 'Type réseau']

for col in colonnes_texte:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encodeurs[col] = le # On le garde pour plus tard !

# On encode aussi la Qualité (Bonne = 0, Mauvaise = 1, Moyenne = 2)
le_qualite = LabelEncoder()
df['Qualite'] = le_qualite.fit_transform(df['Qualite'])
encodeurs['Qualite'] = le_qualite

# 4. Séparer X (input) et y (output)
X = df[['Opérateur', 'Quartier', 'Type réseau', 'Download (Mbps)', 'Upload (Mbps)', 'Latence (ms)', 'Jitter (ms)', 'Loss (%)']]
y = df['Qualite']

# Séparer en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalisation (Mettre toutes les valeurs numériques à la même échelle)
print("Normalisation des données...")
scaler = StandardScaler()
# On normalise seulement les colonnes numériques
colonnes_num = ['Download (Mbps)', 'Upload (Mbps)', 'Latence (ms)', 'Jitter (ms)', 'Loss (%)']
X_train[colonnes_num] = scaler.fit_transform(X_train[colonnes_num])
X_test[colonnes_num] = scaler.transform(X_test[colonnes_num])

# 6. Modélisation : Réseau de neurones (MLPClassifier)
print("Entraînement du modèle...")
# On crée un petit réseau de neurones avec 2 couches cachées
modele = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
 
modele.fit(X_train, y_train) # Entraînement

# Évaluation
y_pred = modele.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"Précision (Accuracy) du modèle : {precision * 100:.2f}%")

# 7. Sauvegarde du modèle, du scaler et des encodeurs
# On utilise "pickle" pour enregistrer ces objets dans des fichiers.
print("Sauvegarde des fichiers...")
with open('modele.pkl', 'wb') as f:
    pickle.dump(modele, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('encodeurs.pkl', 'wb') as f:
    pickle.dump(encodeurs, f)

print("Entraînement terminé avec succès ! Les fichiers .pkl sont créés.")