# cross-validation-fire-detection
Classificação de ocorrência de incêndios utilizando Regressão Logística, padronização de variáveis e validação cruzada com múltiplas métricas de avaliação.

Fire Detection with Cross-Validation

Projeto de Machine Learning desenvolvido para avaliar a ocorrência de incêndios a partir de variáveis obtidas por sensores, utilizando um modelo de Regressão Logística e técnicas de validação cruzada (Cross-Validation).

🎯 Objetivo

O objetivo do projeto é utilizar técnicas de validação cruzada para avaliar a performance de um modelo de classificação na previsão da ocorrência de um incêndio com base nas variáveis fornecidas pela base de dados.

O problema é tratado como uma classificação binária, em que:

0 → ausência de incêndio
1 → ocorrência de incêndio
🗃️ Base de dados

O projeto utiliza uma base de dados de detecção de fumaça e incêndio obtida a partir de sensores IoT.

Antes da modelagem, foram realizadas algumas verificações e etapas de preparação dos dados:

análise dos tipos das variáveis;
renomeação de algumas colunas;
remoção de uma coluna sem utilidade para o modelo;
verificação de valores ausentes;
verificação de dados duplicados;
análise da distribuição da variável dependente.

A variável Fire_Alarm apresentou uma distribuição de aproximadamente:

28,53% → ausência de incêndio;
71,46% → ocorrência de incêndio.

Neste primeiro momento, optou-se por manter a distribuição original dos dados, sem aplicação de técnicas de balanceamento.

🤖 Modelo utilizado

Foi utilizada a Regressão Logística, escolhida por ser um modelo adequado para problemas de classificação binária.

Além disso, como as variáveis preditoras são numéricas, foi realizada a padronização dos dados utilizando StandardScaler.

O modelo foi estruturado em um Pipeline, garantindo que a padronização fosse realizada corretamente dentro do processo de validação cruzada.

Pipeline
Dados
  ↓
StandardScaler
  ↓
Logistic Regression
  ↓
Predição
  ↓
Avaliação
🔄 Validação Cruzada

Para avaliar a capacidade de generalização do modelo, foi utilizada a técnica de K-Fold Cross-Validation com:

5 folds
shuffle=True
random_state=5

A utilização da validação cruzada permite avaliar o desempenho do modelo em diferentes partições dos dados, reduzindo a dependência de uma única divisão entre treinamento e teste.

📊 Métricas de avaliação

Foram utilizadas cinco métricas:

Accuracy
Precision
Recall
F1-score
ROC AUC
Resultados
Métrica	Resultado médio
Accuracy	98,96%
Precision	99,30%
Recall	99,24%
F1-score	99,27%
ROC AUC	99,90%

A acurácia dos cinco folds apresentou valores muito próximos, variando aproximadamente entre 98,83% e 99,04%, indicando que o desempenho do modelo se manteve consistente entre as diferentes partições dos dados.

O recall de 99,24% indica que o modelo conseguiu identificar corretamente a grande maioria dos casos de ocorrência de incêndio.

O ROC AUC de 99,90% indica uma excelente capacidade de discriminação entre as classes.

🛠️ Tecnologias utilizadas
Python
Pandas
Scikit-learn
Regressão Logística
StandardScaler
Pipeline
K-Fold Cross-Validation
📦 Bibliotecas

As principais bibliotecas utilizadas no projeto foram:

import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
▶️ Como executar

Clone este repositório:

git clone https://github.com/SEU-USUARIO/fire-detection-cross-validation.git

Entre na pasta:

cd fire-detection-cross-validation

Instale as dependências:

pip install -r requirements.txt

Execute o script:

python fire_detection_cross_validation.py

Para executar o projeto, a base de dados utilizada pelo código deve estar disponível no diretório esperado pelo script.

📌 Principais conclusões

A validação cruzada apresentou resultados elevados e consistentes para o modelo de Regressão Logística.

O modelo apresentou desempenho superior a 98% nas principais métricas, além de um ROC AUC de 99,90%, demonstrando excelente capacidade de distinguir situações de ocorrência e ausência de incêndio.

Apesar dos resultados positivos, a distribuição original da variável resposta é desigual. Como etapa futura, pode ser interessante investigar o impacto de técnicas de balanceamento e comparar o desempenho com outros algoritmos de classificação.

🚀 Próximos passos

Como possíveis melhorias para o projeto:

testar técnicas de balanceamento das classes;
comparar a Regressão Logística com outros algoritmos de classificação;
realizar ajuste de hiperparâmetros;
analisar a importância e influência das variáveis preditoras;
avaliar a matriz de confusão;
comparar diferentes estratégias de validação;
investigar possíveis problemas de overfitting ou data leakage;
testar o modelo em dados externos.
👩‍💻 Autora

Ana Carolina Martins Pereira

Projeto desenvolvido para estudos de Machine Learning, classificação e validação cruzada.

Ana Carolina Pereira

Projeto desenvolvido para estudos de Machine Learning, classificação e validação cruzada.
