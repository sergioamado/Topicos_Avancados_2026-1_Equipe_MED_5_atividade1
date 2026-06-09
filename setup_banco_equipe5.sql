-- SETUP DO BANCO DE DADOS - EQUIPA 5 (MULTI-TENANT)
-- Disciplina: Tópicos Avançados Engenharia de Software

-- TABELA DE MEMBROS (Chave principal para isolar os dados)
CREATE TABLE IF NOT EXISTS membros (
    id_membro SERIAL PRIMARY KEY,
    nome VARCHAR(100) UNIQUE NOT NULL
);

-- Tabela para Questões de Múltipla Escolha (MCQ - M2)
CREATE TABLE IF NOT EXISTS atividade1_mcq (
    id_mcq SERIAL PRIMARY KEY,
    id_membro INT REFERENCES membros(id_membro) ON DELETE CASCADE,
    modelo VARCHAR(100) NOT NULL,
    pergunta TEXT NOT NULL,
    predicao_do_modelo TEXT,
    gabarito_correto TEXT,
    score INT CHECK (score IN (0, 1))
);

-- Tabela para Questões Abertas (Open - M1)
CREATE TABLE IF NOT EXISTS atividade1_open (
    id_open SERIAL PRIMARY KEY,
    id_membro INT REFERENCES membros(id_membro) ON DELETE CASCADE,
    modelo VARCHAR(100) NOT NULL,
    id_pergunta_original VARCHAR(50),
    pergunta TEXT NOT NULL,
    resposta_ia TEXT,
    gabarito_esperado TEXT
);

-- ATIVIDADE 2: AVALIAÇÃO DO JUIZ E INSIGHTS
CREATE TABLE IF NOT EXISTS atividade2_insights (
    id_insight SERIAL PRIMARY KEY,
    id_membro INT REFERENCES membros(id_membro) ON DELETE CASCADE,
    id_pergunta VARCHAR(50),
    modelo VARCHAR(100) NOT NULL,
    nota_juiz INT CHECK (nota_juiz >= 1 AND nota_juiz <= 5),
    gabarito TEXT,
    resposta_ia TEXT,
    justificativa_juiz TEXT,
    classificacao_erro VARCHAR(255),
    insight_clinico TEXT
);