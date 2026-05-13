import psycopg2
import re
from llama_cpp import Llama

# --- CONFIGURAÇÕES ---
DB_CONFIG = {"dbname": "atividade", "user": "atividade", "password": "atividade@2026", "host": "localhost"}
CAMINHO_MODELO_LOCAL = "./modelos/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
NOME_MODELO_JUIZ = "Llama-3-8B-Local"

print("1. Carregando o Juiz na GPU Local (Isso pode levar uns segundos)...")
# Carrega o modelo usando 100% da RTX 3060
llm = Llama(model_path=CAMINHO_MODELO_LOCAL, n_gpu_layers=-1, n_ctx=4096, verbose=False)

conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = False
cursor = conn.cursor()

try:
    cursor.execute("SELECT id_modelo FROM modelos WHERE nome_modelo = %s;", (NOME_MODELO_JUIZ,))
    resultado = cursor.fetchone()
    if resultado:
        id_juiz = resultado[0]
    else:
        cursor.execute("INSERT INTO modelos (nome_modelo, parametro_precisao) VALUES (%s, 'Local/INT4') RETURNING id_modelo;", (NOME_MODELO_JUIZ,))
        id_juiz = cursor.fetchone()[0]
        conn.commit()

    cursor.execute("""
        SELECT r.id_resposta, p.enunciado, p.resposta_ouro, r.texto_resposta 
        FROM respostas_atividade_1 r 
        JOIN perguntas p ON r.id_pergunta = p.id_pergunta 
        WHERE r.id_resposta NOT IN (SELECT id_resposta_ativa1 FROM avaliacoes_juiz);
    """)
    respostas_pendentes = cursor.fetchall()
    print(f"-> Encontradas {len(respostas_pendentes)} respostas para julgar.\n")

    for id_resp, pergunta, gabarito, resp_ia in respostas_pendentes:
        prompt_sistema = """[PERSONA]
You are a senior cardiologist specialized in international clinical guidelines (AHA/SBC). Evaluate the clinical conduct proposed by an AI system.

[RUBRIC]
1: CRITICAL ERROR. Recommends dangerous conduct, lethal dosage, or ignores classic vital signs.
2: Technically correct conclusion, but omits vital safety steps or mandatory exams.
3: Correct answer, aligned with standard conduct, but lacks long-term management details.
4: Very good answer, follows clinical guidelines and shows good pathophysiological reasoning.
5: Perfect answer, identical or superior to the Gold Standard in clarity and pharmacological precision.

[OUTPUT INSTRUCTIONS]
Be strict. Provide your verdict EXACTLY in this format:
REASONING: <detailed technical justification>
SCORE: <just the number from 1 to 5>"""

        prompt_usuario = f"[CONTEXT]\nQuestion: {pergunta}\nGold Standard: {gabarito}\nAI Answer to evaluate: {resp_ia}"

        try:
            resposta = llm.create_chat_completion(
                messages=[{"role": "system", "content": prompt_sistema}, {"role": "user", "content": prompt_usuario}],
                max_tokens=250, # Espaço para o juiz explicar
                temperature=0.1
            )
            resposta_juiz = resposta["choices"][0]["message"]["content"]

            match_reasoning = re.search(r'REASONING:\s*(.*?)\s*SCORE:', resposta_juiz, re.IGNORECASE | re.DOTALL)
            match_score = re.search(r'SCORE:\s*(\d)', resposta_juiz, re.IGNORECASE)

            if match_reasoning and match_score:
                reasoning = match_reasoning.group(1).strip()
                score = int(match_score.group(1))
            else:
                reasoning = "Falha no parse: " + resposta_juiz
                score = 1

            cursor.execute("""
                INSERT INTO avaliacoes_juiz (id_resposta_ativa1, id_modelo_juiz, nota_atribuida, chain_of_thought)
                VALUES (%s, %s, %s, %s);
            """, (id_resp, id_juiz, score, reasoning))
            
            # Commit a cada 10 respostas para garantir que os dados sejam salvos
            conn.commit() 
            print(f"Resposta ID {id_resp} avaliada! Nota: {score}")

        except Exception as e_local:
            print(f"Erro local ao julgar Resposta {id_resp}: {e_local}")

    print("\n[SUCESSO] Tribunal Local encerrado! Todas as avaliações salvas.")

except Exception as e:
    conn.rollback()
    print(f"[ERRO] Falha geral: {e}")

finally:
    cursor.close()
    conn.close()