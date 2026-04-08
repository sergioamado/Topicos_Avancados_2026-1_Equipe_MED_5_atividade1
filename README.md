# Atividade Avaliativa 1: Curadoria de Datasets e Inferência Básica com LLMs

**Repositório Oficial - Equipe 5 (Domínio Médico)**
**Disciplina:** Tópicos Avançados (2026.1)

## 👥 Equipe 5 (Medicina)
* Sergio Santana dos Santos


🎥 **[[LINK PARA O VÍDEO ([[(https://youtu.be/B3e6lc6I8GQ)]]**

---

## 🎯 Visão Geral do Projeto
Este repositório contém os scripts, datasets e resultados da nossa imersão inicial com Modelos de Linguagem de Grande Escala (LLMs) quantizados para rodar no ambiente local e aplicados ao Domínio Médico. O foco da atividade foi a curadoria especializada, a inferência local e a avaliação crítica das respostas geradas por IA contra padrões-ouro estabelecidos por especialistas.

Trabalhamos com dois subconjuntos de dados:
* **Dataset M1 (Itaymanes K-QA):** Questões abertas baseadas em casos reais com respostas em texto livre (*free-form answer*).
  
* **Dataset M2 (USMLE):** Questões de múltipla escolha focadas no exame de licenciamento médico dos EUA.
  
Para:
* Curadoria de dados
* Inferência local com múltiplos modelos
* Avaliação quantitativa e qualitativa
* Classificação de complexidade clínica
---

## 🛠️ Pré-requisitos e Ferramentas

Para executar os scripts deste repositório, recomendamos a seguinte configuração de ambiente:

1. **IDE (Ambiente de Desenvolvimento):**
   * Recomendamos o uso do **Visual Studio Code (VS Code)**. 
   * 🔗 [Download do VS Code aqui](https://code.visualstudio.com/)

2. **Downloads dos Modelos (Hugging Face):**
   * Os modelos devem ser baixados no formato GGUF e salvos dentro de uma pasta chamada `modelos/` na raiz do projeto.
   * 🔗 **Llama-3 (8B):** [Meta-Llama-3-8B-Instruct.Q4_K_M.gguf](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)
   * 🔗 **Mistral (7B):** [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
   * 🔗 **Phi-3 (3.8B):** [Phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)

---

## ⚙️ Arquitetura Técnica e Configuração de Inferência

Para garantir a máxima performance e privacidade na manipulação dos dados médicos, optamos por rodar os modelos 100% localmente. 

* **Hardware Base:** Ambiente Ubuntu 24.04 rodando em uma NVIDIA RTX 3060 (12GB VRAM).
* **Motor de Inferência:** Utilizamos a biblioteca `llama.cpp` (via `llama-cpp-python`) compilada nativamente com suporte a CUDA (`-DGGML_CUDA=on`). Isso nos permitiu carregar todas as camadas dos modelos na VRAM (`n_gpu_layers=-1`), acelerando drasticamente o processamento.

---
📂 Datasets Utilizados
🔹 Dataset M1 – Questões Abertas
Fonte: K-QA (Itaymanes)
Contém:
Question
Free_form_answer (padrão-ouro)
Must_have
Nice_to_have
Sources
ICD_10_diag
🔹 Dataset M2 – Múltipla Escolha
Fonte: USMLE
Contém questões objetivas com gabarito oficial
---

## 📊 Métricas de Avaliação (Quantitativas e Qualitativas)

Devido à natureza crítica do domínio médico, a avaliação não pode se basear apenas na exatidão das palavras, mas sim no rigor semântico e clínico. Implementamos um pipeline de múltiplas métricas:

### 1. Dataset M1 (Questões Abertas)
* **BERTScore:** Avaliação semântica quantitativa. Utilizamos o modelo `roberta-large` para comparar a intenção e o significado da resposta gerada contra o gabarito padrão-ouro, superando as limitações de métricas lexicais como BLEU.
* **F1-Token:** Avaliação de precisão e recall baseada na exatidão exata das palavras.
* **Desvio de Similaridade:** Cálculo da divergência entre as respostas dos três modelos para analisar diferentes "linhas de raciocínio" clínico.
* **LLM-as-a-Judge (Avaliação Qualitativa):** Utilizamos o Llama-3 atuando como um "Juiz Cego". O modelo avaliou cada par de respostas (Padrão-Ouro vs. Gerada) e atribuiu uma nota de **1 a 5** para a precisão médica e integridade da informação gerada.

### 2. Dataset M2 (Múltipla Escolha)
* **Acurácia Estrita:** Utilização de expressões regulares (Regex) e injeção de *system prompts* restritivos para forçar a extração de uma única letra (A-E), permitindo o cálculo exato de acertos contra o gabarito oficial.
* **Taxa de Concordância:** Métrica binária (Unânime/Divergente) para avaliar o consenso dos três modelos diante do mesmo caso clínico.

---

🧠 Classificação de Dificuldade (Curadoria)

As questões foram classificadas em:

Triagem → Casos simples e diretos
Generalista → Conhecimento clínico básico
Especialista → Exige protocolos e análise clínica
Expert → Casos complexos ou raros
🤝 Ensemble de Modelos

Cada questão é avaliada por três modelos:

LLaMA
Mistral
Phi-3

A classificação final é definida por voto majoritário.

---

📊 Resultados Gerados
**Excel consolidado com:
* Respostas dos modelos
* Classificação por modelo
* BERTScore
* Voto final
**Gráficos:
* ranking_modelos.png
* score_por_classe.png

---

📈 Saídas
M1_RESULTADO_COMPLETO.xlsx
ranking_modelos.png
score_por_classe.png

---

🧾 Conclusão

O uso de múltiplos modelos em ensemble, combinado com métricas semânticas, permitiu uma avaliação robusta da qualidade das respostas e da complexidade clínica das questões.

---

## 🚀 Como Reproduzir os Experimentos

1. Clone este repositório.
2. Instale as dependências listadas no `requirements.txt` (certifique-se de ter os compiladores C++ e o toolkit da NVIDIA instalados se desejar usar aceleração por GPU):

   CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements.txt
---
3. 📊 Pipeline do Projeto
🔹 1. 1_pipeline_m1.py
Geração de respostas para questões abertas
🔹 2. 2_juiz_m1.py
Avaliação qualitativa (LLM-as-a-Judge)
🔹 3. 3_pipeline_m2.py
Resolução de questões objetivas (USMLE)
🔹 4. 4_classificacao_m1.py
Classificação de dificuldade usando ensemble de LLMs
Geração de métricas e gráficos
---
