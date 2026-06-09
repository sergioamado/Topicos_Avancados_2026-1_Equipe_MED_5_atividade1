import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv, find_dotenv

print("-" * 50)
print("📥 EXPORTAÇÃO DE DADOS PARA ANÁLISE (EQUIPE 5)")
print("-" * 50)

load_dotenv(find_dotenv())

try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    print("[OK] Conectado ao PostgreSQL.")
except Exception as e:
    print(f"[ERRO] Falha no banco de dados: {e}")
    exit()

# Consulta para Questões Abertas (Open)
query_open = """
    SELECT 
        a3.id_membro, a3.modelo, a3.id_pergunta_original, r3.pergunta,
        COALESCE(a2.nota_juiz, 0) AS nota_sem_rag,
        a3.nota_juiz AS nota_com_rag,
        r3.resposta_ia_com_rag, a3.chain_of_thought AS justificativa_juiz
    FROM atividade3_avaliacoes_juiz a3
    JOIN atividade3_respostas_rag r3 ON a3.id_resposta_rag = r3.id_resposta_rag
    INNER JOIN atividade1_open a1 ON a3.id_pergunta_original = a1.id_pergunta_original AND a3.id_membro = a1.id_membro AND a3.modelo = a1.modelo
    LEFT JOIN atividade2_insights a2 ON a3.id_pergunta_original = a2.id_pergunta AND a3.id_membro = a2.id_membro AND a3.modelo = a2.modelo
"""

# Consulta para Múltipla Escolha (MCQ)
query_mcq = """
    SELECT 
        a3.id_membro, a3.modelo, a3.id_pergunta_original, r3.pergunta,
        COALESCE(a2.nota_juiz, 0) AS nota_sem_rag,
        a3.nota_juiz AS nota_com_rag,
        r3.resposta_ia_com_rag, a3.chain_of_thought AS justificativa_juiz
    FROM atividade3_avaliacoes_juiz a3
    JOIN atividade3_respostas_rag r3 ON a3.id_resposta_rag = r3.id_resposta_rag
    INNER JOIN atividade1_mcq a1 ON a3.id_pergunta_original = CAST(a1.id_mcq AS VARCHAR) AND a3.id_membro = a1.id_membro AND a3.modelo = a1.modelo
    LEFT JOIN atividade2_insights a2 ON a3.id_pergunta_original = a2.id_pergunta AND a3.id_membro = a2.id_membro AND a3.modelo = a2.modelo
"""

print("\nExtraindo dados...")
df_open = pd.read_sql(query_open, conn)
df_mcq = pd.read_sql(query_mcq, conn)

df_open.to_csv("dados_completos_abertas.csv", index=False, encoding="utf-8")
df_mcq.to_csv("dados_completos_mcq.csv", index=False, encoding="utf-8")

print(f"[SUCESSO] Foram exportadas {len(df_open)} linhas para 'dados_completos_abertas.csv'")
print(f"[SUCESSO] Foram exportadas {len(df_mcq)} linhas para 'dados_completos_mcq.csv'")

conn.close()