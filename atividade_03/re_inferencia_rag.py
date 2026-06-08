import os
import glob
import psycopg2
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from llama_cpp import Llama 

# CONFIGURAÇÕES INICIAIS
load_dotenv("../.env") 

PASTA_MODELOS = "../modelos"
PASTA_CHROMA_DB = "./chroma_db_local"
NOME_COLECAO = "diretrizes_medicas"

print("-" * 50)
print("🤖 RE-INFERÊNCIA MÉDICA COM RAG - EQUIPE 5")
print("-" * 50)

# CONEXÃO COM OS BANCOS DE DADOS
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
    print(f"[ERRO] Falha no PostgreSQL: {e}")
    exit()

# SELEÇÃO DINÂMICA DE MEMBRO
cursor.execute("SELECT id_membro, nome FROM membros ORDER BY id_membro")
membros_cadastrados = cursor.fetchall()

if not membros_cadastrados:
    print("[ERRO] Nenhum membro encontrado no banco de dados.")
    exit()

print("\n👤 Selecione o membro que está executando a inferência:")
for membro in membros_cadastrados:
    print(f"[{membro[0]}] - {membro[1].title()}")

while True:
    try:
        escolha = int(input("\nDigite o número correspondente ao seu nome: "))
        if escolha in [m[0] for m in membros_cadastrados]:
            id_membro = escolha
            nome_membro = next(m[1] for m in membros_cadastrados if m[0] == escolha)
            print(f"[OK] Membro selecionado: {nome_membro.title()}")
            break
        else:
            print("[AVISO] Opção inválida. Tente novamente.")
    except ValueError:
        print("[AVISO] Por favor, digite apenas números.")

# SELEÇÃO DO TIPO DE QUESTÃO (OPEN OU MCQ)
print("\n📚 Qual banco de questões você deseja usar no RAG agora?")
print("[1] - Questões Abertas (Open - M1)")
print("[2] - Múltipla Escolha (MCQ - M2)")

while True:
    try:
        escolha_tipo = int(input("\nDigite 1 ou 2: "))
        if escolha_tipo in [1, 2]:
            tipo_questao = "Open" if escolha_tipo == 1 else "MCQ"
            print(f"[OK] Modo selecionado: {tipo_questao}")
            break
        else:
            print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Por favor, digite 1 ou 2.")

# SELEÇÃO DINÂMICA DO MODELO FÍSICO (.GGUF)
arquivos_modelos = glob.glob(os.path.join(PASTA_MODELOS, "*.gguf"))

if not arquivos_modelos:
    print(f"\n[ERRO] Nenhum modelo .gguf encontrado na pasta '{PASTA_MODELOS}'.")
    exit()

print("\n🧠 Selecione o arquivo do modelo que será usado agora:")
for i, caminho in enumerate(arquivos_modelos):
    print(f"[{i}] - {os.path.basename(caminho)}")

while True:
    try:
        escolha_modelo = int(input("\nDigite o número do modelo: "))
        if 0 <= escolha_modelo < len(arquivos_modelos):
            caminho_modelo_gguf = arquivos_modelos[escolha_modelo]
            print(f"[OK] Arquivo selecionado: {os.path.basename(caminho_modelo_gguf)}")
            break
        else:
            print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Por favor, digite apenas números.")

# ASSOCIAÇÃO COM O NOME NO BANCO DE DADOS
# Busca modelos tanto na tabela Open quanto MCQ
cursor.execute("""
    SELECT modelo FROM atividade1_open WHERE id_membro = %s
    UNION
    SELECT modelo FROM atividade1_mcq WHERE id_membro = %s
""", (id_membro, id_membro))
modelos_banco = cursor.fetchall()

print("\n🏷️  Selecione o nome oficial desse modelo no Banco de Dados:")
for i, mod in enumerate(modelos_banco):
    print(f"[{i}] - {mod[0]}")

while True:
    try:
        escolha_nome = int(input("\nDigite o número do nome oficial: "))
        if 0 <= escolha_nome < len(modelos_banco):
            nome_modelo_db = modelos_banco[escolha_nome][0]
            print(f"[OK] Nome oficial definido como: {nome_modelo_db}")
            break
        else:
            print("[AVISO] Opção inválida.")
    except ValueError:
        print("[AVISO] Por favor, digite apenas números.")

# Conectando ao ChromaDB
chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA_DB)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
colecao_rag = chroma_client.get_collection(name=NOME_COLECAO, embedding_function=emb_fn)
print("\n[OK] Conectado ao ChromaDB (RAG Base).")

# CARREGANDO O MODELO LOCAL
print(f"\nCarregando o modelo local na memória de vídeo/RAM...")
try:
    llm = Llama(model_path=caminho_modelo_gguf, n_ctx=4096, verbose=False)
except Exception as e:
    print(f"[ERRO] Falha ao carregar o modelo: {e}")
    exit()

# EXECUÇÃO DO PIPELINE RAG (AGORA RODA TUDO SEM LIMITES)
if escolha_tipo == 1:
    cursor.execute("""
        SELECT id_pergunta_original, pergunta 
        FROM atividade1_open 
        WHERE id_membro = %s AND modelo = %s;
    """, (id_membro, nome_modelo_db))
else:
    # Para o MCQ, mapeamos o id_mcq como id_pergunta_original para salvar na mesma tabela de resultados
    cursor.execute("""
        SELECT CAST(id_mcq AS VARCHAR), pergunta 
        FROM atividade1_mcq 
        WHERE id_membro = %s AND modelo = %s;
    """, (id_membro, nome_modelo_db))

perguntas = cursor.fetchall()

if not perguntas:
    print(f"\n[AVISO] Nenhuma pergunta encontrada para o modelo {nome_modelo_db} neste dataset.")
    exit()

print(f"\nIniciando inferência RAG para todas as {len(perguntas)} perguntas encontradas...")

for id_pergunta, texto_pergunta in perguntas:
    print(f"\n[Processando] Pergunta ID: {id_pergunta}")
    
    # RECUPERAÇÃO (RAG)
    resultados = colecao_rag.query(
        query_texts=[texto_pergunta],
        n_results=3 
    )
    
    documentos_recuperados = resultados['documents'][0]
    fontes_recuperadas = resultados['metadatas'][0]
    
    contexto_texto = ""
    for i, doc in enumerate(documentos_recuperados):
        fonte = fontes_recuperadas[i]['fonte']
        pagina = fontes_recuperadas[i]['pagina']
        contexto_texto += f"--- Fragmento {i+1} (Fonte: {fonte}, Pág: {pagina}) ---\n{doc}\n\n"
    
    print("  -> Contexto recuperado com sucesso dos PDFs!")

    # PROMPT ENGINEERING (Adaptado para Open ou MCQ)
    if escolha_tipo == 1:
        instrucao_especifica = "Answer the medical question thoroughly."
    else:
        instrucao_especifica = "Since this is a multiple-choice question, explicitly state the correct option letter (A, B, C, D, or E) and briefly explain why."

    prompt_aumentado = f"""You are a highly skilled clinical medical AI. 
{instrucao_especifica} Base your answer EXCLUSIVELY on the updated clinical guidelines provided in the context below. 
Do not use outdated knowledge. If the context does not contain the exact answer, use your clinical reasoning combined with the context.

MEDICAL GUIDELINES CONTEXT:
{contexto_texto}

QUESTION:
{texto_pergunta}

ANSWER:"""

    # GERAÇÃO (INFERÊNCIA DO MODELO)
    print("  -> Gerando resposta baseada nas diretrizes (Aguarde...)")
    resposta = llm(
        prompt_aumentado,
        max_tokens=500,
        stop=["QUESTION:", "MEDICAL GUIDELINES CONTEXT:"],
        echo=False
    )
    
    texto_resposta_ia = resposta['choices'][0]['text'].strip()
    
    # SALVANDO NO BANCO DE DADOS
    cursor.execute("""
        INSERT INTO atividade3_respostas_rag 
        (id_membro, modelo, id_pergunta_original, pergunta, contexto_recuperado, resposta_ia_com_rag)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (id_membro, nome_modelo_db, id_pergunta, texto_pergunta, contexto_texto, texto_resposta_ia))
    
    conn.commit()
    print("  -> Salvo no PostgreSQL com sucesso!")

print("\n[PROCESSO CONCLUÍDO] Todas as respostas com RAG foram geradas e salvas!")
cursor.close()
conn.close()