from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# --- CHARGEMENT DES FICHIERS SAUVEGARDÉS ---
with open('modele.pkl', 'rb') as f:
    modele = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encodeurs.pkl', 'rb') as f:
    encodeurs = pickle.load(f)

# Route pour afficher la page Web (HTML)
@app.route('/')
def home():
    return render_template('index.html')

# Route pour recevoir les données et faire la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Récupérer les données envoyées en JSON
        data = request.get_json()

        # 2. Convertir en DataFrame (comme lors de l'entraînement)
        df_input = pd.DataFrame([data])

        # 3. Encoder le texte en nombres
        for col in ['Opérateur', 'Quartier', 'Type réseau']:
            # On utilise l'encodeur sauvegardé pour transformer le texte
            df_input[col] = encodeurs[col].transform(df_input[col])

        # 4. Normaliser les nombres
        colonnes_num = ['Download (Mbps)', 'Upload (Mbps)', 'Latence (ms)', 'Jitter (ms)', 'Loss (%)']
        df_input[colonnes_num] = scaler.transform(df_input[colonnes_num])

        # 5. Faire la prédiction
        prediction_num = modele.predict(df_input)[0]

        # 6. Re-transformer le nombre en texte (ex: 0 -> "Bonne")
        prediction_texte = encodeurs['Qualite'].inverse_transform([prediction_num])[0]

        # 7. Renvoyer le résultat
        return jsonify({'prediction': prediction_texte})

    except Exception as e:
        return jsonify({'erreur': str(e)})

if __name__ == '__main__':
    # 0.0.0.0 permet d'y accéder depuis un autre PC sur le même réseau WiFi
    app.run(host='0.0.0.0', port=5000, debug=True)                  