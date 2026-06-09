import os
import glob
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST", "localhost"), 
    "port": os.getenv("DB_PORT", "5432")       
}

def conectar_banco():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"[ERRO] Não foi possível conectar ao banco de dados: {e}")
        print("Verifique se o seu arquivo .env está configurado corretamente com DB_NAME, DB_USER, etc.")
        exit()

# IDENTIFICAÇÃO DO MEMBRO
print("-" * 50)
print("🩺 SISTEMA DE INGESTÃO DE DADOS - EQUIPE 5")
print("-" * 50)
nome_digitado = input("Digite o seu nome e sobrenome: ").strip().lower()

conn = conectar_banco()
cursor = conn.cursor()

cursor.execute("SELECT id_membro FROM membros WHERE LOWER(nome) = %s", (nome_digitado,))
resultado = cursor.fetchone()

if resultado:
    id_membro = resultado[0]
    print(f"\n[OK] Bem-vindo de volta! Seu ID de membro é: {id_membro}")
else:
    cursor.execute("INSERT INTO membros (nome) VALUES (%s) RETURNING id_membro", (nome_digitado,))
    id_membro = cursor.fetchone()[0]
    conn.commit()
    print(f"\n[NOVO CADASTRO] Membro registrado com sucesso! Seu ID é: {id_membro}")

# IMPORTAÇÃO DA ATIVIDADE 1 (MCQ E OPEN)
pasta_atv1 = "atividade_01"

print(f"\nProcurando arquivos na pasta '{pasta_atv1}'...")

if not os.path.exists(pasta_atv1):
    print(f"[AVISO] A pasta '{pasta_atv1}' não existe. Crie a pasta e coloque os seus CSVs lá.")
else:
    arquivos_mcq = glob.glob(os.path.join(pasta_atv1, "MCQ_*.csv"))
    for arquivo in arquivos_mcq:
        nome_modelo = os.path.basename(arquivo).replace("MCQ_", "").replace(".csv", "")
        df = pd.read_csv(arquivo)
        coluna_predicao = [col for col in df.columns if "prediction" in col.lower()]
        if coluna_predicao:
            df.rename(columns={coluna_predicao[0]: "predicao_do_modelo"}, inplace=True)
        
        dados_insercao = []
        for _, row in df.iterrows():
            dados_insercao.append((id_membro, nome_modelo, row['question'], row.get('predicao_do_modelo', ''), row.get('correct', ''), row.get('score', 0)))
        
        query = """INSERT INTO atividade1_mcq (id_membro, modelo, pergunta, predicao_do_modelo, gabarito_correto, score) 
                   VALUES (%s, %s, %s, %s, %s, %s)"""
        execute_batch(cursor, query, dados_insercao)
        conn.commit()
        print(f"  -> Inseridas {len(df)} linhas do arquivo {os.path.basename(arquivo)}")

    arquivos_open = glob.glob(os.path.join(pasta_atv1, "Open_*.csv"))
    for arquivo in arquivos_open:
        nome_modelo = os.path.basename(arquivo).replace("Open_", "").replace(".csv", "")
        df = pd.read_csv(arquivo)
        
        dados_insercao = []
        for _, row in df.iterrows():
            dados_insercao.append((id_membro, nome_modelo, str(row.get('id', '')), row.get('question', ''), row.get('answer', ''), row.get('must_have_score', '')))
        
        query = """INSERT INTO atividade1_open (id_membro, modelo, id_pergunta_original, pergunta, resposta_ia, gabarito_esperado) 
                   VALUES (%s, %s, %s, %s, %s, %s)"""
        execute_batch(cursor, query, dados_insercao)
        conn.commit()
        print(f"  -> Inseridas {len(df)} linhas do arquivo {os.path.basename(arquivo)}")

# IMPORTAÇÃO DA ATIVIDADE 2
pasta_atv2 = "atividade_02"
print(f"\nProcurando arquivos na pasta '{pasta_atv2}'...")

if not os.path.exists(pasta_atv2):
    print(f"[AVISO] A pasta '{pasta_atv2}' não existe. Crie a pasta e coloque os seus CSVs lá.")
else:
    arquivos_insights = glob.glob(os.path.join(pasta_atv2, "insights_*.csv"))
    for arquivo in arquivos_insights:
        df = pd.read_csv(arquivo)
        
        dados_insercao = []
        for _, row in df.iterrows():
            dados_insercao.append((
                id_membro, str(row.get('ID_Pergunta', '')), row.get('Modelo', ''), 
                row.get('Nota', 0), row.get('Gabarito', ''), row.get('Resposta_IA', ''), 
                row.get('Justificativa_Juiz', ''), row.get('Classificacao_Erro', ''), row.get('Insight_Clinico', '')
            ))
        
        query = """INSERT INTO atividade2_insights 
                   (id_membro, id_pergunta, modelo, nota_juiz, gabarito, resposta_ia, justificativa_juiz, classificacao_erro, insight_clinico) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        execute_batch(cursor, query, dados_insercao)
        conn.commit()
        print(f"  -> Inseridas {len(df)} linhas do arquivo {os.path.basename(arquivo)}")

print("\n[PROCESSO CONCLUÍDO] Todos os seus dados foram carregados e amarrados ao seu perfil com segurança!")

cursor.close()
conn.close()