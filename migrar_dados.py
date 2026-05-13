import psycopg2
import json

# Configuração da Ligação à Base de Dados
conn = psycopg2.connect(
    dbname="atividade",
    user="atividade",
    password="atividade@2026",
    host="localhost"
)
conn.autocommit = False
cursor = conn.cursor()

try:
    print("1. Criando o Esquema Relacional Oficial...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS modelos (
            id_modelo SERIAL PRIMARY KEY,
            nome_modelo VARCHAR(100) NOT NULL,
            versao VARCHAR(50),
            parametro_precisao VARCHAR(20)
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id_dataset SERIAL PRIMARY KEY,
            nome_dataset VARCHAR(100) NOT NULL,
            dominio VARCHAR(50) NOT NULL
        );

        CREATE TABLE IF NOT EXISTS perguntas (
            id_pergunta SERIAL PRIMARY KEY,
            id_dataset INTEGER REFERENCES datasets(id_dataset),
            enunciado TEXT NOT NULL,
            resposta_ouro TEXT NOT NULL,
            metadados JSONB
        );

        CREATE TABLE IF NOT EXISTS respostas_atividade_1 (
            id_resposta SERIAL PRIMARY KEY,
            id_pergunta INTEGER REFERENCES perguntas(id_pergunta),
            id_modelo INTEGER REFERENCES modelos(id_modelo),
            texto_resposta TEXT NOT NULL,
            tempo_inferencia_ms FLOAT,
            data_geracao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS avaliacoes_juiz (
            id_avaliacao SERIAL PRIMARY KEY,
            id_resposta_ativa1 INTEGER REFERENCES respostas_atividade_1(id_resposta),
            id_modelo_juiz INTEGER REFERENCES modelos(id_modelo),
            nota_atribuida INTEGER CHECK (nota_atribuida BETWEEN 1 AND 5),
            chain_of_thought TEXT NOT NULL,
            data_avaliacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    print("2. Inserindo Datasets Iniciais...")
    cursor.execute("INSERT INTO datasets (nome_dataset, dominio) VALUES ('K-QA (Abertas)', 'Médico') RETURNING id_dataset;")
    id_m1 = cursor.fetchone()[0]
    
    cursor.execute("INSERT INTO datasets (nome_dataset, dominio) VALUES ('USMLE (Múltipla Escolha)', 'Médico') RETURNING id_dataset;")
    id_m2 = cursor.fetchone()[0]

    def obter_id_modelo(nome_modelo):
        cursor.execute("SELECT id_modelo FROM modelos WHERE nome_modelo = %s;", (nome_modelo,))
        res = cursor.fetchone()
        if res: return res[0]
        cursor.execute("INSERT INTO modelos (nome_modelo, parametro_precisao) VALUES (%s, 'N/A') RETURNING id_modelo;", (nome_modelo,))
        return cursor.fetchone()[0]

    def processar_tabelas(prefixo, id_dataset):
        cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '{prefixo}%';")
        tabelas = [row[0] for row in cursor.fetchall()]

        for tabela in tabelas:
            print(f"   -> Lendo tabela: {tabela}")
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s;", (tabela,))
            todas_colunas = [row[0] for row in cursor.fetchall()]

            # Mapeamento adaptativo de colunas (ignora maiúsculas/minúsculas)
            col_id = next((c for c in todas_colunas if c.lower() in ['id_questao', 'identificador', 'question']), None)
            col_perg = next((c for c in todas_colunas if c.lower() in ['pergunta', 'question']), None)
            col_gab = next((c for c in todas_colunas if c.lower() in ['gabarito', 'resposta_correta', 'answer']), None)
            
            # Pega em todas as colunas que começam por 'resposta_' (mas que não sejam o gabarito)
            colunas_respostas = [c for c in todas_colunas if c.lower().startswith('resposta_') and c.lower() != 'resposta_correta']

            if not (col_perg and col_gab and colunas_respostas):
                print(f"      [AVISO] Colunas essenciais em falta na {tabela}. A ignorar...")
                continue

            colunas_select = f'"{col_perg}", "{col_gab}"'
            if col_id:
                colunas_select = f'"{col_id}", ' + colunas_select
            
            colunas_select += ', ' + ', '.join([f'"{col}"' for col in colunas_respostas])

            cursor.execute(f'SELECT {colunas_select} FROM "{tabela}";')
            linhas = cursor.fetchall()
            
            for i_row, row in enumerate(linhas):
                offset = 1 if col_id else 0
                val_id = row[0] if col_id else i_row
                val_perg = row[offset]
                val_gab = row[offset+1]

                if not val_gab or str(val_gab) == 'nan': continue
                
                metadados = json.dumps({"id_original": val_id, "origem": tabela})
                cursor.execute("INSERT INTO perguntas (id_dataset, enunciado, resposta_ouro, metadados) VALUES (%s, %s, %s, %s) RETURNING id_pergunta;", (id_dataset, str(val_perg), str(val_gab), metadados))
                id_pergunta = cursor.fetchone()[0]
                
                for i, col in enumerate(colunas_respostas):
                    # Limpa o nome do modelo (ex: 'resposta_gpt' vira 'gpt')
                    nome_modelo_limpo = col.replace('Resposta_', '').replace('resposta_', '')
                    id_mod = obter_id_modelo(nome_modelo_limpo)
                    resposta = row[offset + 2 + i]
                    
                    if resposta and str(resposta) != 'nan':
                        cursor.execute("INSERT INTO respostas_atividade_1 (id_pergunta, id_modelo, texto_resposta, tempo_inferencia_ms) VALUES (%s, %s, %s, 0.0);", (id_pergunta, id_mod, str(resposta)))

    print("\n3. Migrando dados do M1 (Questões Abertas)...")
    processar_tabelas('resultado_m1', id_m1)

    print("\n4. Migrando dados do M2 (Múltipla Escolha)...")
    processar_tabelas('resultado_m2', id_m2)

    conn.commit()
    print("\n[SUCESSO ABSOLUTO] A Base de Dados Relacional está unificada! Dados de TODOS os membros da equipa foram salvos.")

except Exception as e:
    conn.rollback()
    print(f"\n[ERRO] Falha na transação. Detalhes: {e}")

finally:
    cursor.close()
    conn.close()