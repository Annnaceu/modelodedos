import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Importar a biblioteca joblib para salvar o modelo

# Carregue os dados
data = np.load('dados_dedos.npy', allow_pickle=True)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo os parâmetros para o GridSearch
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Treinando o modelo com Grid Search
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Avaliando o modelo
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy:.2f}')

# Salvando o modelo treinado
joblib.dump(best_model, 'modelo_contagem_dedos.pkl')  # Salva o modelo
