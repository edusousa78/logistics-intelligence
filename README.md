
# NexaLog Brasil | Logistics Intelligence

## Inteligência e Desempenho Logístico

Projeto de Ciência de Dados e Business Intelligence voltado à análise histórica de entregas do comércio eletrônico brasileiro.

A solução utiliza dados públicos do Brazilian E-Commerce Public Dataset by Olist para investigar atrasos logísticos, acompanhar indicadores operacionais e apoiar a análise gerencial.

## Objetivo de negócio

Transformar registros históricos de pedidos em informações claras para acompanhamento do desempenho logístico.

O dashboard permite:

- Monitorar volumes de entregas.
- Analisar taxas históricas de atraso.
- Comparar períodos operacionais.
- Investigar prazos logísticos.
- Consultar pedidos e exportar resultados.
- Consultar os resultados de um experimento preditivo.

## Base de dados

Fonte: Brazilian E-Commerce Public Dataset by Olist.

O conjunto original contém aproximadamente 100 mil pedidos e informações relacionadas a clientes, produtos, vendedores, pagamentos e entregas.

O dashboard publicado utiliza um recorte histórico de validação de março e abril de 2018, com 13.703 pedidos.

## Indicadores do dashboard

| Indicador | Resultado |
|---|---:|
| Pedidos analisados | 13.703 |
| Entregas dentro do prazo | 12.041 |
| Entregas atrasadas | 1.662 |
| Taxa histórica de atraso | 12,13% |

Os indicadores correspondem exclusivamente ao período analisado e não representam operações em tempo real.

## Experimento de Machine Learning

Foi desenvolvido e avaliado um modelo de classificação para investigar a possibilidade de identificar pedidos com risco de atraso.

O experimento utilizou separação temporal dos dados e avaliação final em um conjunto de teste reservado.

### Resultados no teste final

| Métrica | Resultado |
|---|---:|
| Pedidos avaliados | 12.516 |
| Taxa de atraso | 4,79% |
| ROC-AUC | 0,6324 |
| Average Precision | 0,0731 |
| Precision@10% | 0,0663 |
| Recall@10% | 0,1386 |
| Lift@10% | 1,3852 |

**Decisão técnica: NO-GO para implantação preditiva operacional.**

O modelo não atingiu o critério mínimo de discriminação definido para o experimento.

Consequentemente, a aplicação final foi posicionada como uma solução de análise histórica e Business Intelligence. O modelo é mantido exclusivamente como artefato de pesquisa e documentação.

## Tecnologias

- Python
- Pandas e NumPy
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebook
- Git e GitHub

## Estrutura do projeto

```text
logistics-intelligence/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── processed/
│       └── dashboard_validation.csv
├── models/
│   ├── logistic_pipeline.joblib
│   └── model_metadata.json
├── notebooks/
│   └── logistics_intelligence.ipynb
├── reports/
│   └── final_holdout_evaluation.json
└── src/
    └── inference.py
```

## Execução local

Clone o repositório e acesse a pasta do projeto.

Crie um ambiente virtual:

```powershell
python -m venv .venv
```

Ative o ambiente no Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Execute o dashboard:

```powershell
python -m streamlit run app.py
```

A aplicação será disponibilizada no endereço local indicado pelo Streamlit.

## Limitações

- Os dados analisados são históricos.
- Os indicadores não representam monitoramento operacional em tempo real.
- O modelo preditivo não foi aprovado para implantação.
- O dashboard não deve ser utilizado como ferramenta automatizada de decisão sobre entregas reais.
- A interpretação dos resultados deve considerar as características e limitações do dataset.

## Fonte dos dados

Brazilian E-Commerce Public Dataset by Olist:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Autor

Luis Sousa

GitHub: https://github.com/edusousa78

Projeto desenvolvido para portfólio profissional em Ciência de Dados.
