# 🩺 ATIVIDADE 1: A Fundação e a "Memória Pura" (Inferência Básica e Curadoria)

## O Objetivo

Descobrir como os modelos de IA se comportam ao responder a perguntas médicas complexas usando apenas os dados com os quais foram treinados originalmente (sua "memória pura"), rodando 100% offline para garantir o sigilo dos dados dos pacientes.

### 1. Os Dados (O Padrão-Ouro)

Utilizamos dois datasets rigorosos:

**M1 (K-QA):** Questões abertas (texto livre) baseadas em casos clínicos reais (ex: "Paciente de 45 anos apresenta dor no peito... qual a conduta?").

**M2 (USMLE):** Questões de múltipla escolha focadas no dificílimo Exame de Licenciamento Médico dos Estados Unidos.

### 2. Os Motores de Geração (Modelos Locais)

Para processar os dados offline, rodamos três modelos de pesos abertos no seu computador (Ubuntu + RTX 3060 12GB VRAM):

* Llama-3 (8B) da Meta.
* Mistral (7B) da Mistral AI.
* Phi-3 (3.8B) da Microsoft.

**Engenharia de Performance:** Usamos a biblioteca llama.cpp compilada nativamente com suporte a CUDA (-DGGML_CUDA=on). Isso permitiu carregar todas as matrizes matemáticas dos modelos diretamente na memória da placa de vídeo (VRAM), acelerando drasticamente o tempo de resposta das predições.

### 3. A Avaliação Matemática e Semântica (Os Modelos Ocultos)

Como avaliar se um texto médico gerado pela IA está correto? Em medicina, uma palavra muda tudo.

**Acurácia Estrita (M2):** Para múltipla escolha, usamos Expressões Regulares (Regex) para forçar o sistema a ler apenas a letra final (A, B, C) e comparar com o gabarito.

**F1-Token (M1):** Mediu a sobreposição exata de palavras entre a resposta da IA e o gabarito.

**BERTScore (O 4º Modelo):** Aqui entrou o roberta-large, um modelo codificador treinado pelo Facebook. Ele não gera texto; ele entende o significado. Ele leu as respostas e calculou a distância semântica. Se o gabarito dizia "Hipertensão" e a IA dizia "Pressão Alta", o RoBERTa percebia que clinicamente era a mesma coisa e dava nota alta.

### 4. O Sistema de Ensemble (Votação)

Criamos um "conselho médico" digital. Colocamos o Llama, o Mistral e o Phi-3 para votar. Se os três acertassem a questão, ela era classificada como Triagem (Fácil). Se divergiam muito, era classificada como Expert (Difícil). A decisão final saía por "Voto Majoritário".

---

# ⚖️ ATIVIDADE 2: A Auditoria Clínica e o Juiz na Nuvem (LLM-as-a-Judge)

## O Objetivo

Abandonar a avaliação puramente matemática e trazer um raciocínio clínico de alto nível para auditar as respostas dos modelos locais. Além disso, sair das planilhas de Excel e criar uma infraestrutura de dados profissional.

### 1. Migração para Banco de Dados Relacional

Scripts em Excel quebram facilmente. Construímos uma infraestrutura robusta utilizando o PostgreSQL.

Criamos o banco bd_equipe5 com um esquema Multi-Tenant (capaz de separar os dados por membro da equipe).

Desenhamos tabelas interligadas por Chaves Estrangeiras (id_membro, id_pergunta_original), garantindo que uma resposta do Mistral nunca se misturasse com o gabarito do Llama.

### 2. O Juiz na Nuvem (O 5º Modelo)

Modelos de 8B parâmetros (locais) são ótimos para gerar ideias, mas não têm a profundidade necessária para julgar outros modelos com precisão cirúrgica.

Conectamo-nos via API à Groq para utilizar o gigantesco Llama-3.3-70B.

Ele assumiu o papel de Juiz Auditor. Enviamos a ele a pergunta, a resposta da IA menor e o gabarito. Ele analisou clinicamente, deu uma nota de 1 a 5 e escreveu o porquê (Chain of Thought).

### 3. Engenharia de Resiliência (Rate Limits e Checkpointing)

Como a API da Groq tem limites severos de uso gratuito (RPM - Requests Per Minute), o script não podia simplesmente falhar no meio do caminho.

Criamos lógicas de try/except com Hibernação (o código pausava automaticamente por 60 segundos ao sentir o bloqueio da API).

Graças ao PostgreSQL, cada nota salva virava um "Checkpoint". Se o computador desligasse, o script recomeçava exatamente da pergunta onde parou (usando cláusulas SQL NOT IN), sem gastar tokens julgando a mesma coisa duas vezes.

### 4. Classificação de Erros e o 1º Dashboard

O Llama-3.3 também foi encarregado de classificar os erros: "Omissão de Dosagem", "Diagnóstico Incorreto", "Conduta Perigosa". Com isso, geramos insights clínicos em .csv e criamos a nossa primeira interface visual (Streamlit) para mostrar o risco real que os modelos locais traziam aos pacientes.

---

# 🚀 ATIVIDADE 3: A Cura das Alucinações e a Prova Científica (RAG)

## O Objetivo

Resolver o problema descoberto na Atividade 2. Os modelos locais estavam alucinando condutas médicas porque estavam com pesos desatualizados (desconheciam o uso moderno de Inibidores de SGLT2 ou Agonistas GLP-1). A missão foi injetar conhecimento atualizado em tempo real sem precisar treinar os modelos de novo.

### 1. A Arquitetura RAG (Retrieval-Augmented Generation)

Em vez de depender da memória do modelo, ensinamo-lo a consultar a literatura médica moderna.

**Os Documentos:** Fornecemos as Diretrizes da ADA 2024 (Diabetes) e AHA 2023 (Doença Coronariana).

**O Banco Vetorial (ChromaDB):** O 6º modelo do nosso projeto entrou em ação: um codificador de Embeddings (all-MiniLM-L6-v2). Ele fatiou os PDFs em "chunks" (pedaços de texto) e transformou-os em vetores (coordenadas 3D).

**A Injeção:** Quando o utilizador fazia a pergunta, o ChromaDB encontrava matematicamente o parágrafo do PDF da ADA/AHA que respondia àquela dúvida. O script juntava esse parágrafo à pergunta e enviava para o modelo local (Llama/Mistral/Phi), forçando-o a usar aquele contexto para responder. Salvemos essas novas respostas na tabela atividade3_respostas_rag.

### 2. O Juiz Resiliente e o Fallback Automático

Precisávamos que o Juiz avaliasse as novas respostas geradas pelo RAG. Mas a Groq tem limites rígidos.

Desenvolvemos uma arquitetura de Fallback Automático de API.

O script tentava enviar a avaliação para o Llama-3.3-70B (Groq). Se a Groq devolvesse um erro 429 Rate Limit ou esgotasse os tokens diários, o nosso código em Python "virava a chave" automaticamente na mesma fração de segundo e transferia o trabalho para o Google Gemini 1.5 Flash (O 7º Modelo do projeto).

O processamento nunca parava. Ele continuou até julgar 100% da base de dados.

### 3. A Comprovação Científica (Estatística)

Em MLOps, sentimento não importa; apenas dados importam.

Fizemos consultas SQL unindo (JOIN) a Atividade 2 (Sem RAG) com a Atividade 3 (Com RAG).

Usamos as bibliotecas Pandas e SciPy para calcular o Coeficiente de Correlação de Spearman.

O que isso provou? Medimos se a injeção do PDF apenas subiu as notas proporcionalmente (Correlação Forte) ou se reescreveu completamente a lógica médica da IA, salvando pacientes de condutas perigosas (Correlação Fraca/Moderada). Comprovamos isso gerando gráficos de impacto (.png) através da biblioteca Matplotlib.

### 4. O Dashboard Final e a Portabilidade (Streamlit e pg_dump)

Para coroar o projeto, precisávamos de uma forma executiva de apresentar esses dados.

Criamos o dashboard_equipe5.py usando Streamlit. Um painel web com separadores que extrai os dados ao vivo do PostgreSQL. Ele permite filtrar por modelo e por médico, ver os gráficos estatísticos e abrir abas expansíveis para ler lado a lado a resposta alucinada (Ativ. 1) e a resposta clinicamente perfeita guiada pelo RAG (Ativ. 3), junto com o comentário do Juiz IA.

Para garantir que esse painel e esses dados pudessem ser usados por qualquer membro da equipe em qualquer lugar do mundo, desenvolvemos scripts de exportação massiva para .csv e utilizamos o comando de segurança pg_dump para fazer o backup completo do banco de dados relacional.

Foram utilizados 7 modelos distintos (geradores, juízes e codificadores), 2 paradigmas de banco de dados (Relacional/SQL e Vetorial/NoSQL) e ferramentas padrão da indústria de Data Science. É um case de portfólio fenomenal.
