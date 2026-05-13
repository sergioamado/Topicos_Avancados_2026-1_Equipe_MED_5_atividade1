import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# 1. Carregar variáveis do arquivo .env
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

print("1. Conectando ao Banco de Dados...")

try:
    # Estabelece a conexão com o PostgreSQL
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Esta query SQL cruza as 4 tabelas para montar um relatório completo!
    query = """
        SELECT 
            p.id_pergunta,
            d.nome_dataset AS dataset,
            m_resp.nome_modelo AS modelo_avaliado,
            p.enunciado AS pergunta,
            p.resposta_ouro AS gabarito,
            r.texto_resposta AS resposta_da_ia,
            m_juiz.nome_modelo AS modelo_juiz,
            a.nota_atribuida AS nota_do_juiz,
            a.chain_of_thought AS justificativa_do_juiz
        FROM avaliacoes_juiz a
        JOIN respostas_atividade_1 r ON a.id_resposta_ativa1 = r.id_resposta
        JOIN perguntas p ON r.id_pergunta = p.id_pergunta
        JOIN datasets d ON p.id_dataset = d.id_dataset
        JOIN modelos m_resp ON r.id_modelo = m_resp.id_modelo
        JOIN modelos m_juiz ON a.id_modelo_juiz = m_juiz.id_modelo
        ORDER BY p.id_pergunta, m_resp.nome_modelo;
    """
    
    print("2. Extraindo as avaliações do Juiz-IA...")
    # O Pandas lê a query e já transforma em um DataFrame (tabela)
    df_avaliacoes = pd.read_sql_query(query, conn)
    
    if df_avaliacoes.empty:
        print("[AVISO] Nenhuma avaliação encontrada no banco. O Juiz já avaliou alguma coisa?")
    else:
        # 3. Salvar em CSV
        nome_arquivo = "avaliacoes_consolidadas_equipe5.csv"
        df_avaliacoes.to_csv(nome_arquivo, index=False, encoding='utf-8')
        
        print(f"\n[SUCESSO] Foram exportadas {len(df_avaliacoes)} avaliações!")
        print(f"-> Arquivo salvo como: {nome_arquivo}")

except Exception as e:
    print(f"[ERRO] Falha ao exportar os dados: {e}")

finally:
    if 'conn' in locals() and not conn.closed:
        conn.close()