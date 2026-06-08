-- SETUP DO BANCO DE DADOS - ATIVIDADE 3 (RAG)
-- Disciplina: Tópicos Avançados ENgenharia de Software

-- TABELA DE RESPOSTAS COM RAG
-- Guarda a nova inferência dos modelos após lerem o PDF
CREATE TABLE IF NOT EXISTS atividade3_respostas_rag (
    id_resposta_rag SERIAL PRIMARY KEY,
    id_membro INT REFERENCES membros(id_membro) ON DELETE CASCADE,
    modelo VARCHAR(100) NOT NULL,
    id_pergunta_original VARCHAR(50),
    pergunta TEXT NOT NULL,
    contexto_recuperado TEXT, -- Guarda exatamente os parágrafos do PDF injetados no prompt
    resposta_ia_com_rag TEXT,
    data_geracao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TABELA DE AVALIAÇÃO DO JUIZ (PÓS-RAG)
-- Guarda a nota do Juiz 70B e a sua avaliação sobre se o PDF ajudou ou atrapalhou
CREATE TABLE IF NOT EXISTS atividade3_avaliacoes_juiz (
    id_avaliacao_rag SERIAL PRIMARY KEY,
    id_membro INT REFERENCES membros(id_membro) ON DELETE CASCADE,
    id_resposta_rag INT REFERENCES atividade3_respostas_rag(id_resposta_rag) ON DELETE CASCADE,
    modelo VARCHAR(100) NOT NULL,
    id_pergunta_original VARCHAR(50),
    nota_juiz INT CHECK (nota_juiz >= 1 AND nota_juiz <= 5),
    chain_of_thought TEXT
);

-- VIEW COMPARATIVA
-- Esta view cruza as tabelas da Atividade 2 com as da Atividade 3 e calcula o ganho
CREATE OR REPLACE VIEW vw_comparativo_rag AS
SELECT 
    r3.id_membro,
    mem.nome AS nome_membro,
    r3.id_pergunta_original,
    r3.modelo,
    a2.nota_juiz AS nota_sem_rag,
    a3.nota_juiz AS nota_com_rag,
    (a3.nota_juiz - a2.nota_juiz) AS evolucao_nota, 
    a3.chain_of_thought AS analise_do_juiz_sobre_o_rag
FROM atividade3_respostas_rag r3
JOIN membros mem ON mem.id_membro = r3.id_membro
JOIN atividade3_avaliacoes_juiz a3 ON a3.id_resposta_rag = r3.id_resposta_rag
LEFT JOIN atividade2_insights a2 
    ON a2.id_membro = r3.id_membro 
    AND a2.id_pergunta = r3.id_pergunta_original 
    AND a2.modelo = r3.modelo;