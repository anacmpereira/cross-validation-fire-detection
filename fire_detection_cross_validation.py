# PROJETO CROSS VALIDATION

# O objetivo desse projeto é utilizar técnicas de validação cruzada (cross-validation) para avaliar a performance
# de um modelo de classificação na previsão da ocorrência de um incêndio com base nas variáveis fornecidas.

# Importando bibliotecas
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Importando base de dados
pd.set_option("display.max_columns", None)
base = pd.read_csv("Cientista de dados M35 - smoke_detection_iot.csv", delimiter = ',')
print(base)

# Verificações iniciais
print(base.dtypes)
base.rename(columns={'Fire Alarm': 'Fire_Alarm', 'Raw H2': 'Raw_H2', 'Raw Ethanol': 'Raw_Ethanol'}, inplace=True) # renomeando algumas colunas
base = base.drop(columns=["Unnamed: 0"]) # removendo coluna que não srá putil para o modelo
print(base.columns)

# Verificando dados ausentes
print(base.isnull().sum()) # sem dados nulos/ausentes

# Verificando dados duplicados
print(base.duplicated().sum()) # sem dados duplicados

# Verificando o balanceamento da variável dependente
print(base["Fire_Alarm"].value_counts())
print(base["Fire_Alarm"].value_counts(normalize=True))

# A classe 0 (ausência de incêndio) é minoritária, representa apenas 28,53% dos dados. A classe 1 (incêndio) representa 71,46%
# dos dados. O balanceamento dos dados não será feito inicialmente, a fim de avaliar o desempenho do modelo com a
# distribuição original das observações. Caso os resultados indiquem dificuldade na identificação da classe minoritária,
# poderá ser testada uma abordagem de balanceamento.

# A regressão logística foi escolhida por ser um modelo adequado para problemas de classificação binária, nos quais a variável
# resposta apresenta duas categorias: ausência (0) ou ocorrência de incêndio (1). Como as variáveis preditoras são numéricas,
# elas podem ser utilizadas pelo modelo após a padronização. A regressão logística permite estimar a probabilidade de ocorrência
# de um incêndio a partir das características observadas. Além disso, por ser um modelo relativamente simples e interpretável,
# possibilita avaliar a influência das variáveis preditoras sobre a ocorrência do evento, sendo uma opção adequada como modelo inicial.

# Separando a base em X e Y
x = base.drop('Fire_Alarm', axis = 1)
y = base['Fire_Alarm']

# Instanciando o modelo de regressão logística
modelo = Pipeline([
    ('padronizacao', StandardScaler()),
    ('regressao_logistica', LogisticRegression(max_iter=1000))
])

# Configurações do cross validation
folds = 5 # definindo o número de divisões
crossvalidation = KFold(n_splits=folds, shuffle=True, random_state=5)

# Aplicando o Cross Validation
resultados = cross_validate(modelo, x, y, cv=crossvalidation, scoring=['accuracy','precision','recall','f1','roc_auc'])

# Resultados
print("Accuracy por fold:", resultados['test_accuracy'])
print("Accuracy média:", resultados['test_accuracy'].mean())

print("Precision média:", resultados['test_precision'].mean())
print("Recall médio:", resultados['test_recall'].mean())
print("F1 médio:", resultados['test_f1'].mean())
print("ROC AUC médio:", resultados['test_roc_auc'].mean())



