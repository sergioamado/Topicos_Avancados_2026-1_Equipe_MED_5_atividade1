import os
import json
import psycopg2
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, RateLimitError

# CONFIGURAÇÕES INICIAIS
load_dotenv(find_dotenv())

print("-" * 50)
print("⚖️  JUIZ IA NA NUVEM - AVALIAÇÃO DO RAG (EQUIPE 5)")
print("-" * 50)

# CONFIGURAÇÃO DOS CLIENTES DAS APIs (GROQ E GEMINI)
try:
    # Cliente Primário (Mais rápido, mas com limite rígido)
    cliente_groq = OpenAI(
        api_key=os.getenv("API_KEY_GROQ"),
        base_url=os.getenv("BASE_URL_GROQ")
    )
    modelo_groq = os.getenv("NOME_MODELO_GROQ", "llama-3.3-70b-versatile")

    # Cliente Secundário (Fallback de emergência)
    cliente_gemini = OpenAI(
        api_key=os.getenv("API_KEY_GEMINI"),
        base_url=os.getenv("BASE_URL_GEMINI")
    )
    modelo_gemini = os.getenv("NOME_MODELO_GEMINI", "gemini-1.5-pro")
except Exception as e:
    print(f"[ERRO] Falha ao configurar as chaves de API: {e}")
    print("Verifique seu arquivo .env!")
    exit()

# CONEXÃO COM O BANCO DE DADOS POSTGRESQL
try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    cursor = conn.cursor()
    print("[OK] Conectado ao PostgreSQL.")
except Exception as e:
    print(f"[ERRO] Falha no banco de dados: {e}")
    exit()

# SELEÇÃO DINÂMICA DE MEMBRO E BANCO DE QUESTÕES
cursor.execute("SELECT id_membro, nome FROM membros ORDER BY id_membro")
membros = cursor.fetchall()

print("\n👤 Selecione o membro para avaliar as respostas:")
for m in membros:
    print(f"[{m[0]}] - {m[1].title()}")

while True:
    try:
        id_membro = int(input("\nDigite o ID do membro: "))
        if id_membro in [m[0] for m in membros]:
            nome_membro = next(m[1] for m in membros if m[0] == id_membro)
            break
        print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Digite apenas números.")

print("\n📚 Qual banco de questões o Juiz deve avaliar agora?")
print("[1] - Questões Abertas (Open - M1)")
print("[2] - Múltipla Escolha (MCQ - M2)")

while True:
    try:
        escolha_tipo = int(input("\nDigite 1 ou 2: "))
        if escolha_tipo in [1, 2]:
            break
        print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Por favor, digite 1 ou 2.")

# SELEÇÃO DO MODELO LOCAL AVALIADO
cursor.execute("SELECT DISTINCT modelo FROM atividade3_respostas_rag WHERE id_membro = %s", (id_membro,))
modelos_disponiveis = cursor.fetchall()

if not modelos_disponiveis:
    print("\n[AVISO] Nenhuma resposta com RAG encontrada para este membro.")
    exit()

print("\n🤖 Qual modelo local você deseja que o Juiz avalie agora?")
for i, mod in enumerate(modelos_disponiveis):
    print(f"[{i}] - {mod[0]}")

while True:
    try:
        escolha_mod = int(input("\nDigite o número correspondente: "))
        if 0 <= escolha_mod < len(modelos_disponiveis):
            modelo_avaliado = modelos_disponiveis[escolha_mod][0]
            break
        print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Digite apenas números.")

# BUSCA AS RESPOSTAS PARA AVALIAR (JÁ PROTEGIDO CONTRA DUPLICAÇÃO PELO "NOT IN")
if escolha_tipo == 1:
    consulta_sql = """
        SELECT r3.id_resposta_rag, r3.id_pergunta_original, r3.pergunta, 
               r3.resposta_ia_com_rag, r3.contexto_recuperado, a1.gabarito_esperado
        FROM atividade3_respostas_rag r3
        JOIN atividade1_open a1 
            ON r3.id_pergunta_original = a1.id_pergunta_original 
            AND r3.id_membro = a1.id_membro AND r3.modelo = a1.modelo
        WHERE r3.id_membro = %s AND r3.modelo = %s
          AND r3.id_resposta_rag NOT IN (SELECT id_resposta_rag FROM atividade3_avaliacoes_juiz)
    """
else:
    consulta_sql = """
        SELECT r3.id_resposta_rag, r3.id_pergunta_original, r3.pergunta, 
               r3.resposta_ia_com_rag, r3.contexto_recuperado, a2.gabarito_correto
        FROM atividade3_respostas_rag r3
        JOIN atividade1_mcq a2 
            ON r3.id_pergunta_original = CAST(a2.id_mcq AS VARCHAR) 
            AND r3.id_membro = a2.id_membro AND r3.modelo = a2.modelo
        WHERE r3.id_membro = %s AND r3.modelo = %s
          AND r3.id_resposta_rag NOT IN (SELECT id_resposta_rag FROM atividade3_avaliacoes_juiz)
    """

cursor.execute(consulta_sql, (id_membro, modelo_avaliado))
respostas_para_avaliar = cursor.fetchall()

if not respostas_para_avaliar:
    print("\n[OK] Todas as respostas deste modelo/tipo já foram avaliadas pelo Juiz! Nenhuma duplicata será gerada.")
    exit()

print(f"\nIniciando avaliação de {len(respostas_para_avaliar)} respostas geradas com RAG...")

# Variáveis de controle de Fallback
api_ativa = "groq"
parar_tudo = False

for linha in respostas_para_avaliar:
    if parar_tudo:
        break

    id_resposta_rag, id_pergunta, pergunta_texto, resposta_rag, contexto_usado, gabarito = linha
    print(f"\n[Avaliando] Pergunta ID: {id_pergunta}...")

    criterio = "Compare a 'Resposta Gerada' com o 'Gabarito Oficial'. Avalie a precisão clínica." if escolha_tipo == 1 else "Verifique rigorosamente se a IA escolheu a alternativa correta do 'Gabarito Oficial'."

    prompt_juiz = f"""Você é um Juiz Médico Especialista (Avaliador de IA).
Sua tarefa é avaliar a resposta dada por uma IA menor a uma questão clínica, baseando-se no Gabarito Oficial e no Contexto (RAG) que a IA usou.

PERGUNTA:
{pergunta_texto}

GABARITO OFICIAL ESPERADO:
{gabarito}

CONTEXTO FORNECIDO À IA (RAG):
{contexto_usado}

RESPOSTA GERADA PELA IA MENOR:
{resposta_rag}

INSTRUÇÕES DE AVALIAÇÃO:
1. {criterio}
2. Verifique se a IA utilizou corretamente o 'Contexto Fornecido' ou se ela o ignorou/alucinou.
3. Atribua uma nota de 1 a 5 (onde 1 é completamente errada/perigosa e 5 é exata e clinicamente precisa).

Retorne SUA AVALIAÇÃO EXCLUSIVAMENTE em formato JSON, com as seguintes chaves:
"chain_of_thought": "Justificativa da nota escrita obrigatoriamente em PORTUGUÊS DO BRASIL (PT-BR). Deve ser MUITO CURTA, DIRETA e OBJETIVA (máximo 2 a 3 frases limitadas ao essencial).",
"nota_juiz": <um número inteiro de 1 a 5>
"""

    # Usamos um While True interno para permitir que o script tente novamente 
    # a MESMA pergunta caso o Groq falhe e troquemos para o Gemini
    while True:
        try:
            # Define quem vai julgar a rodada atual
            cliente_atual = cliente_groq if api_ativa == "groq" else cliente_gemini
            modelo_atual = modelo_groq if api_ativa == "groq" else modelo_gemini
            nome_juiz = "Llama-3.3 (Groq)" if api_ativa == "groq" else "Gemini-1.5 (Google)"

            response = cliente_atual.chat.completions.create(
                model=modelo_atual,
                messages=[{"role": "user", "content": prompt_juiz}],
                temperature=0.1, 
                response_format={"type": "json_object"} 
            )
            
            # Formata a resposta limpa para JSON (remove possiveis blocos de markdown ```json)
            resultado_str = response.choices[0].message.content.strip()
            if resultado_str.startswith("```json"):
                resultado_str = resultado_str.replace("```json", "").replace("```", "").strip()
                
            avaliacao = json.loads(resultado_str)
            nota = avaliacao.get("nota_juiz", 1)
            cot = f"[{nome_juiz}] " + avaliacao.get("chain_of_thought", "Não fornecido.")
            
            # Salva no banco
            cursor.execute("""
                INSERT INTO atividade3_avaliacoes_juiz 
                (id_membro, id_resposta_rag, modelo, id_pergunta_original, nota_juiz, chain_of_thought)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (id_membro, id_resposta_rag, modelo_avaliado, id_pergunta, nota, cot))
            
            conn.commit()
            print(f"  -> Sucesso! Julgado por {nome_juiz} - Nota atribuída: {nota}/5")
            break # Sai do loop de tentativa e vai para a próxima pergunta!

        except RateLimitError:
            if api_ativa == "groq":
                print("\n⚠️  [AVISO] Groq esgotou os tokens! Transferindo processamento para o Gemini Pro de forma automática...")
                api_ativa = "gemini" # Vira a chave
                continue # Volta para o início do While para tentar a mesma pergunta de novo
            else:
                print("\n🛑 [ALERTA] Ambas as APIs (Groq e Gemini) esgotaram seus tokens!")
                print("O script foi pausado com segurança. Volte amanhã para continuar de onde parou.")
                parar_tudo = True
                break # Sai do while interno
        
        except Exception as e:
            print(f"  -> [ERRO] Falha ao avaliar a pergunta {id_pergunta}. Pulando para a próxima. Detalhes: {e}")
            conn.rollback()
            break # Falha por outro motivo (ex: json mal formatado), pula a pergunta

print("\n[PROCESSO CONCLUÍDO] Fim das avaliações na nuvem. Todos os dados foram salvos no banco de dados.")
cursor.close()
conn.close()