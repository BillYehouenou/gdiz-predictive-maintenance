import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. Chargement d'un dataset d'exemple
data = {
    'air_temp': [300, 310, 305, 298, 302],
    'process_temp': [310, 320, 315, 308, 312],
    'speed': [1500, 1400, 1550, 1600, 1450],
    'torque': [40, 50, 45, 35, 42],
    'tool_wear': [100, 200, 150, 50, 120],
    'failure': [0, 1, 0, 0, 0] # 1 = Panne
}
df = pd.DataFrame(data)

# 2. Features et Cible
X = df.drop('failure', axis=1)
y = df['failure']

# 3. Entraînement
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 4. Sauvegarde : on l'enregistre dans le dossier app/ pour que l'API y ait accès
joblib.dump(model, 'app/model.joblib')
print("Modèle entraîné et sauvegardé dans app/model.joblib")