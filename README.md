# 🩺 Projeto de Inteligência Artificial Aplicada ao Domínio Médico

## Disciplina: Tópicos Avançados (2026.1)

### Equipe 5 (MED 5)

**Integrante:**

* Sergio Santana dos Santos

🎥 **Vídeo de Apresentação**
https://youtu.be/B3e6lc6I8GQ

---

# 📚 Documentação

Este projeto possui documentação complementar para facilitar a compreensão dos dados utilizados durante as etapas de curadoria, inferência, avaliação e auditoria clínica.

| Documento                                     | Descrição                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [Dicionário de Dados](./dicionario_dados.pdf) | Estrutura das tabelas, atributos, relacionamentos e campos utilizados ao longo do projeto. |

---

# 📑 Índice

* Visão Geral
* Estrutura do Projeto
* Pré-requisitos
* Modelos Utilizados
* Atividade 01 – Curadoria e Inferência Local
* Atividade 02 – LLM-as-a-Judge e Auditoria Clínica
* Atividade 03 – RAG e Avaliação de Desempenho Clínico
* Como Executar
* Resultados Gerados
* Conclusão

---

# 🎯 Visão Geral

Este repositório contém o desenvolvimento completo de três atividades práticas envolvendo Modelos de Linguagem de Grande Escala (LLMs) aplicados ao domínio médico.

O projeto abrange:

* Curadoria especializada de datasets médicos;
* Inferência local utilizando modelos quantizados;
* Avaliação quantitativa e qualitativa das respostas;
* Auditoria clínica automatizada utilizando LLM-as-a-Judge;
* Implementação de Retrieval-Augmented Generation (RAG);
* Construção de dashboards analíticos para apoio à decisão.

Todos os experimentos foram desenvolvidos com foco em privacidade, reprodutibilidade e avaliação rigorosa da qualidade clínica das respostas geradas por Inteligência Artificial.

---

# 📂 Estrutura do Projeto

```text
.
├── atividade01/
│   ├── scripts da atividade
│   ├── datasets utilizados
│   └── resultados gerados
│
├── atividade02/
│   ├── scripts de auditoria clínica
│   ├── integração PostgreSQL
│   ├── avaliação LLM-as-a-Judge
│   └── dashboard Streamlit
│
├── atividade03/
│   ├── documentos/
│   │   ├── Standards of Care in Diabetes 2024 (ADA)
│   │   └── Guideline for Chronic Coronary Disease 2023 (AHA/ACC)
│   │
│   ├── scripts RAG
│   ├── avaliação comparativa
│   └── dashboard analítico
│
├── chromadb_local/
│   └── base vetorial utilizada pelo pipeline RAG
│
├── modelos/
│   ├── Meta-Llama-3
│   ├── Mistral
│   └── Phi-3
│
├── dicionario_dados.pdf
└── README.md
```

---

# 🛠️ Pré-requisitos

## Hardware Utilizado

* Ubuntu 24.04
* NVIDIA RTX 3060 12GB
* CUDA Toolkit
* Python 3.11+

## Ambiente Recomendado

* Visual Studio Code
* PostgreSQL
* ChromaDB
* Streamlit

---

# 🤖 Modelos Utilizados

Os modelos foram executados localmente utilizando `llama.cpp` e `llama-cpp-python`.

| Modelo           | Tamanho |
| ---------------- | ------- |
| Llama-3 Instruct | 8B      |
| Mistral Instruct | 7B      |
| Phi-3 Mini       | 3.8B    |

Todos os modelos devem ser armazenados na pasta:

```text
modelos/
```

---

# 📊 Atividade 01 — Curadoria e Inferência Local

## Objetivos

* Curadoria de datasets médicos;
* Inferência local com múltiplos modelos;
* Avaliação quantitativa;
* Classificação da complexidade clínica.

## Datasets Utilizados

### Dataset M1 — Questões Abertas (K-QA)

Contém:

* Question
* Free_form_answer
* Must_have
* Nice_to_have
* Sources
* ICD_10_diag

### Dataset M2 — USMLE

Contém:

* Questões de múltipla escolha;
* Alternativas;
* Gabarito oficial.

---

## Métricas Utilizadas

### Questões Abertas

* BERTScore
* F1-Token
* Desvio de Similaridade
* LLM-as-a-Judge

### Questões Objetivas

* Acurácia Estrita
* Taxa de Concordância

---

## Classificação de Complexidade

As questões foram classificadas em:

* Triagem
* Generalista
* Especialista
* Expert

---

## Ensemble de Modelos

Cada questão foi avaliada por:

* Llama-3
* Mistral
* Phi-3

A classificação final foi definida por voto majoritário.

---

# 🏥 Atividade 02 — LLM-as-a-Judge e Auditoria Clínica

Nesta etapa foi implementado um pipeline completo de MLOps para avaliação automática de respostas médicas utilizando um modelo de grande escala na nuvem.

## Modelo Juiz

* Llama-3.3-70B
* Groq API

## Funcionalidades

* Integração PostgreSQL;
* Avaliação clínica automática;
* Controle de Rate Limit;
* Classificação automática de erros médicos;
* Dashboard analítico;
* Benchmarking entre modelos.

---

## Scripts Desenvolvidos

| Script                       | Função                 |
| ---------------------------- | ---------------------- |
| setup_banco.py               | Criação das tabelas    |
| inserir_respostas.py         | Carga das respostas    |
| gerar_csv_finais.py          | Geração dos CSVs       |
| juiz_nuvem.py                | Avaliação clínica      |
| exportar_avaliacoes.py       | Exportação consolidada |
| gerar_insights_csv_sergio.py | Extração de insights   |
| app_auditoria_final.py       | Dashboard Streamlit    |

---

# 🧠 Atividade 03 — RAG e Avaliação de Desempenho Clínico

Esta atividade implementa Retrieval-Augmented Generation (RAG) para enriquecer o conhecimento dos modelos locais utilizando diretrizes médicas oficiais.

---

## Base de Conhecimento

Os documentos utilizados encontram-se em:

```text
atividade03/documentos/
```

### Diretriz ADA

**Standards of Care in Diabetes — 2024**

Utilizada para atualização de protocolos relacionados a:

* Diabetes Tipo 1
* Diabetes Tipo 2
* GLP-1
* Controle cardiovascular

### Diretriz AHA/ACC

**2023 Guideline for Chronic Coronary Disease**

Utilizada para atualização de protocolos relacionados a:

* Cardiologia
* Doença coronária crônica
* Betabloqueadores
* Tratamentos cardiovasculares

---

## Base Vetorial

Os embeddings gerados são armazenados em:

```text
chromadb_local/
```

---

## Pipeline RAG

### 1. re_inferencia_rag.py

Geração de novas respostas utilizando contexto recuperado do ChromaDB.

### 2. juiz_nuvem_rag.py

Avaliação clínica das respostas enriquecidas com RAG.

### 3. analise_estatistica_rag.py

Análise comparativa entre:

* Sem RAG
* Com RAG

Gerando:

* Correlação de Spearman
* Gráficos comparativos

### 4. dashboard_equipe5.py

Dashboard interativo para exploração dos resultados.

### 5. exportar_dados.py

Exportação consolidada dos dados para CSV.

---

# 🚀 Como Executar

## Instalar Dependências

```bash
pip install -r requirements.txt
```

Para aceleração via GPU:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements.txt
```

---

## Atividade 01

Executar os scripts localizados em:

```text
atividade01/
```

---

## Atividade 02

Executar os scripts localizados em:

```text
atividade02/
```

---

## Atividade 03

Executar os scripts localizados em:

```text
atividade03/
```

---

# 📈 Resultados Gerados

## Planilhas

* M1_RESULTADO_COMPLETO.xlsx
* avaliacoes_consolidadas_equipe5.csv

## Gráficos

* ranking_modelos.png
* score_por_classe.png
* gráficos comparativos do RAG

## Dashboards

* app_auditoria_final.py
* dashboard_equipe5.py

---

# 🎓 Conclusão

O projeto demonstrou a viabilidade do uso de modelos de linguagem executados localmente para aplicações clínicas, combinando:

* Curadoria especializada;
* Avaliação quantitativa;
* Auditoria clínica automatizada;
* Recuperação de conhecimento externo via RAG;
* Dashboards analíticos para suporte à decisão.

A utilização de múltiplos modelos em ensemble, associada a métricas semânticas e avaliação clínica automatizada, permitiu uma análise robusta da qualidade das respostas e dos riscos associados à utilização de IA no domínio médico.
