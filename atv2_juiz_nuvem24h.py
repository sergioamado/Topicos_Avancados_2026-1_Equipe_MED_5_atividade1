import os
import psycopg2
import re
import time
from datetime import datetime, timedelta
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

client = Groq(api_key=GROQ_API_KEY)

print("1. Iniciando o Juiz em Nuvem (Com gerenciamento de Cota Diária)...")


while True:
    # A cada ciclo reconecta ao banco
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cursor = conn.cursor()

    try:
        # Verifica se o juiz está cadastrado
        cursor.execute("SELECT id_modelo FROM modelos WHERE nome_modelo = %s;", (NOME_MODELO_JUIZ,))
        res_mod = cursor.fetchone()
        if res_mod:
            id_juiz = res_mod[0]
        else:
            cursor.execute("INSERT INTO modelos (nome_modelo, parametro_precisao) VALUES (%s, 'Nuvem/API') RETURNING id_modelo;", (NOME_MODELO_JUIZ,))
            id_juiz = cursor.fetchone()[0]
            conn.commit()

        # Busca apenas as que ainda não têm avaliação
        cursor.execute("""
            SELECT r.id_resposta, p.enunciado, p.resposta_ouro, r.texto_resposta 
            FROM respostas_atividade_1 r 
            JOIN perguntas p ON r.id_pergunta = p.id_pergunta 
            WHERE r.id_resposta NOT IN (SELECT id_resposta_ativa1 FROM avaliacoes_juiz);
        """)
        respostas_pendentes = cursor.fetchall()

        # Se a lista estiver vazia acabou
        if len(respostas_pendentes) == 0:
            print("\n[SUCESSO ABSOLUTO] Todas as respostas do banco foram avaliadas!")
            break # encerra o programa

        print(f"\n-> Retomando o trabalho: {len(respostas_pendentes)} respostas na fila.\n")

        limite_diario_atingido = False

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

                # Salva a avaliação com sucesso
                cursor.execute("""
                    INSERT INTO avaliacoes_juiz (id_resposta_ativa1, id_modelo_juiz, nota_atribuida, chain_of_thought)
                    VALUES (%s, %s, %s, %s);
                """, (id_resp, id_juiz, score, reasoning))
                
                conn.commit()
                print(f"[OK] Resposta {id_resp} avaliada e salva. Score: {score}")

            except Exception as e_api:
                error_msg = str(e_api)
                
                # Se esbarrar no limite de tokens ou requisições
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    print(f"\n[ALERTA] Cota de 100.000 tokens diários esgotada na Resposta {id_resp}.")
                    limite_diario_atingido = True
                    break # Interrompe a avaliação imediatamente
                else:
                    print(f"[ERRO API] Falha na Resposta {id_resp}: {e_api}")
                    conn.rollback()
                    continue # Se for outro erro de conexão, pula e tenta a próxima

        
        if limite_diario_atingido:
            # FECHA o banco evitar timeout
            cursor.close()
            conn.close()
            
            
            segundos_espera = (24 * 60 * 60) + (10 * 60)
            hora_retorno = datetime.now() + timedelta(seconds=segundos_espera)
            
            print(f"\n[HIBERNAÇÃO] O banco foi salvo e fechado em segurança.")
            print(f"-> O robô vai dormir por 24 horas e 10 minutos.")
            print(f"-> ELE ACORDARÁ AUTOMATICAMENTE EM: {hora_retorno.strftime('%d/%m/%Y às %H:%M:%S')}")
            print("Pode ir descansar! Não feche este terminal e certifique-se que o PC não vai suspender.\n")
            
            time.sleep(segundos_espera)

        else:

            break

    except Exception as e:
        print(f"[ERRO GERAL FATAL] {e}")
        conn.rollback()
        break

    finally:
        
        if not conn.closed:
            cursor.close()
            conn.close()