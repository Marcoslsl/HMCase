# HMCase

- Objetivo Principal: Responder à algumas questões de negócio.
- Objetivos secundários:
    - Seguir boas práticas do desenvolvimento de software.
    - Análise estatística
    - Machine learning

## Como vizualizar o projeto

Vá até o arquivo Main.ipynb e dentro do notebook vai estar todo desenvolvimento
do projeto

## Metodologia e passos

1. Estração de dados
2. EDA
3. Machine Learning (regressão e clusterização)
4. Questões de negócio.

## Estrutura do Projeto

- src: Diretório principal
- data: armazenar o .csv do banco de dados
- infra: Conecção com o banco de dados
- model: modelo de machine learning
- tests: testes unitários
- utils: funções auxiliáres, classes e pipelines que foram encapsulados.
```
├── install.sh
├── Main.ipynb
├── README.md
├── requirements.txt
└── src
    ├── data
    │   └── data.csv
    ├── infra
    │   ├── connection.py
    │   └── __init__.py
    ├── __init__.py
    ├── model
    │   └── randomForestModel.pkl
    ├── tests
    │   ├── test_machine_learning.py
    │   ├── test_preprocess.py
    │   └── test_utils.py
    └── utils
        ├── __init__.py
        ├── machine_learning.py
        ├── preprocess.py
        └── utils.py

```