import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error as root_mean_squared_error

# Carregar os dados
url = "top_insta_influencers_data.csv"
data = pd.read_csv(url)

# Função para transformar valores na coluna 'country' em faixas numéricas baseadas em continentes
def transform_country(country):
    if country in ["Brazil", "Colombia", "Uruguay"]:
        return 1
    elif country in ["United States", "Canada", "Puerto Rico", "Anguilla", "British Virgin Islands"]:
        return 2
    elif country in ["Spain", "United Kingdom", "Netherlands", "France", "Switzerland", "Sweden", "Czech Republic", "Germany", "Russia"]:
        return 3
    elif country in ["China", "India", "Indonesia"]:
        return 4
    elif country in ["Australia"]:
        return 5
    elif country in ["Turkey", "United Arab Emirates"]:
        return 6
    elif country in ["Côte d'Ivoire"]:
        return 7
    elif country in ["Mexico"]:
        return 8
    else:
        return 0

# Aplicar a função de transformação à coluna 'country'
data['country'] = data['country'].apply(transform_country)

# Função para converter valores com sufixos k, M, m, b
def convert_to_numeric(value):
    if pd.isna(value):
        return np.nan
    value = value.replace('%', '')  # Remover símbolo de porcentagem
    if 'k' in value:
        return float(value.replace('k', '')) * 1e3
    elif 'M' in value or 'm' in value:
        return float(value.replace('M', '').replace('m', '')) * 1e6
    elif 'b' in value or 'B' in value:
        return float(value.replace('b', '').replace('B', '')) * 1e9
    else:
        return float(value)

# Aplicar a função para converter as colunas relevantes
for col in ['followers', 'avg_likes', '60_day_eng_rate', 'total_likes']:
    data[col] = data[col].apply(convert_to_numeric)

# Remover linhas com valores faltantes após a conversão
data = data.dropna(subset=['followers', 'avg_likes', '60_day_eng_rate', 'total_likes'])

# Selecionando variáveis preditoras e alvo
X = data[['followers', 'avg_likes', '60_day_eng_rate', 'total_likes']]
y = data['influence_score']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementação do kNN com GridSearchCV para otimização dos hiperparâmetros
param_grid = {'n_neighbors': range(1, 31), 'metric': ['euclidean', 'manhattan']}
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Melhor modelo e parâmetros
best_knn = grid_search.best_estimator_
print("Melhores parâmetros: ", grid_search.best_params_)

# Avaliação do modelo
y_pred = best_knn.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred, squared=False)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Gráficos

# Distribuição dos Seguidores
plt.figure(figsize=(12, 6))
sns.histplot(data['followers'], bins=20, kde=True)
plt.title('Distribuição de Followers')
plt.show()

# Distribuição da Média de Curtidas
plt.figure(figsize=(12, 6))
sns.histplot(data['avg_likes'], bins=20, kde=True)
plt.title('Distribuição de Avg Likes')
plt.show()

# Relação entre Seguidores e Média de Curtidas
plt.figure(figsize=(16, 9))
plt.scatter(data['followers'], data['avg_likes'], color='blue', label='Média de curtidas')
plt.xlabel('Seguidores')
plt.ylabel('Média de curtidas')
plt.title('Relação entre Seguidores e Média de Curtidas')
plt.legend()
plt.show()

# Impacto da Taxa de Engajamento de 60 Dias na Média de Curtidas
plt.figure(figsize=(16, 9))
plt.scatter(data['60_day_eng_rate'], data['avg_likes'], color='red', label='Média de curtidas')
plt.xlabel('Taxa de Engajamento de 60 Dias (%)')
plt.ylabel('Média de Curtidas')
plt.title('Impacto da Taxa de Engajamento de 60 Dias na Média de Curtidas')
plt.legend()
plt.show()

# Matriz de Correlação
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()
