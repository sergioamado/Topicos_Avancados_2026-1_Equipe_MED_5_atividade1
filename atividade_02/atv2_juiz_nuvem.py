import os
import psycopg2
import re
import time
from dotenv import load_dotenv
from groq import Groq

# Carregar variáveis do arquivo .env
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOME_MODELO_JUIZ = "Llama-3.3-70B-Groq"

print("1. Conectando ao Banco e Configurando o Juiz (Nuvem via .env)...")

client = Groq(api_key=GROQ_API_KEY)
conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = False
cursor = conn.cursor()

try:
    cursor.execute("SELECT id_modelo FROM modelos WHERE nome_modelo = %s;", (NOME_MODELO_JUIZ,))
    res_mod = cursor.fetchone()
    if res_mod:
        id_juiz = res_mod[0]
    else:
        cursor.execute("INSERT INTO modelos (nome_modelo, parametro_precisao) VALUES (%s, 'Nuvem/API') RETURNING id_modelo;", (NOME_MODELO_JUIZ,))
        id_juiz = cursor.fetchone()[0]
        conn.commit()

    # Só busca as que NÃO têm avaliação (se o script parar, ele continua de onde parou)
    cursor.execute("""
        SELECT r.id_resposta, p.enunciado, p.resposta_ouro, r.texto_resposta 
        FROM respostas_atividade_1 r 
        JOIN perguntas p ON r.id_pergunta = p.id_pergunta 
        WHERE r.id_resposta NOT IN (SELECT id_resposta_ativa1 FROM avaliacoes_juiz);
    """)
    respostas_pendentes = cursor.fetchall()
    print(f"-> {len(respostas_pendentes)} respostas encontradas na fila para julgamento.\n")

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
REASONING: <detailed technical justification in Portuguese>
SCORE: <just the number from 1 to 5>"""

        prompt_usuario = f"[CONTEXT]\nQuestion: {pergunta}\nGold Standard: {gabarito}\nAI Answer to evaluate: {resp_ia}"

        # Loop de repetição (Retry) para lidar com a Groq
        while True:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1
                )
                
                resposta_texto = chat_completion.choices[0].message.content

                match_reasoning = re.search(r'REASONING:\s*(.*?)\s*SCORE:', resposta_texto, re.IGNORECASE | re.DOTALL)
                match_score = re.search(r'SCORE:\s*(\d)', resposta_texto, re.IGNORECASE)

                if match_reasoning and match_score:
                    reasoning = match_reasoning.group(1).strip()
                    score = int(match_score.group(1))
                else:
                    reasoning = f"Erro de formatação na resposta: {resposta_texto[:200]}..."
                    score = 1 

                cursor.execute("""
                    INSERT INTO avaliacoes_juiz (id_resposta_ativa1, id_modelo_juiz, nota_atribuida, chain_of_thought)
                    VALUES (%s, %s, %s, %s);
                """, (id_resp, id_juiz, score, reasoning))
                
                conn.commit()
                print(f"[OK] Resposta {id_resp} julgada. Score: {score}")
                
                break 

            except Exception as e_api:
                error_msg = str(e_api)
                
                # Se for erro 429 (Limite de Tokens atingido)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    # Tenta ler o tempo que a Groq mandou esperar 
                    match = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', error_msg)
                    if match:
                        minutos = int(match.group(1)) if match.group(1) else 0
                        segundos = float(match.group(2))
                        # Tempo exato + 10 segundos de garantia
                        espera_segundos = (minutos * 60) + segundos + 10 
                    else:
                        espera_segundos = 480 # Se não conseguir ler, aguarda 8 minutos por padrão

                    print(f"\n[PAUSA AUTOMÁTICA] Limite de tokens da Groq atingido.")
                    print(f"O script vai dormir por {espera_segundos / 60:.1f} minutos e continuar sozinho depois. Não feche o terminal!\n")
                    
                    time.sleep(espera_segundos)
                
                else:
                    print(f"[ERRO API] Falha grave no ID {id_resp}: {e_api}")
                    conn.rollback()
                    break

    print("\n[PROCESSO CONCLUÍDO] O Juiz em nuvem finalizou todas as avaliações da fila.")

except Exception as e:
    print(f"[ERRO GERAL] {e}")
    conn.rollback()

finally:
    cursor.close()
    conn.close()