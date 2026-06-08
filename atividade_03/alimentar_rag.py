import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CONFIGURAÇÕES DO BANCO VETORIAL
print("-" * 50)
print("INGESTÃO DE CONHECIMENTO RAG - EQUIPE 5")
print("-" * 50)

PASTA_DOCUMENTOS = "./documentos"
PASTA_CHROMA_DB = "./chroma_db_local"
NOME_COLECAO = "diretrizes_medicas"

print("1. Conectando ao Banco Vetorial ChromaDB...")
os.makedirs(PASTA_DOCUMENTOS, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA_DB)
print("   -> Carregando modelo de Embeddings (Multilingual)...")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

colecao = chroma_client.get_or_create_collection(
    name=NOME_COLECAO,
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"} 
)

# LEITURA E CHUNKING DOS PDFs
arquivos_pdf = glob.glob(os.path.join(PASTA_DOCUMENTOS, "*.pdf"))

if not arquivos_pdf:
    print(f"\n[ERRO FATAL] Nenhum PDF encontrado na pasta '{PASTA_DOCUMENTOS}'.")
    print("Por favor, coloque as diretrizes médicas lá dentro e rode novamente.")
    exit()

print(f"\n2. Foram encontrados {len(arquivos_pdf)} documento(s). Iniciando leitura...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

textos_para_salvar = []
metadados = []
ids = []
contador_id = 0

for arquivo in arquivos_pdf:
    nome_arquivo = os.path.basename(arquivo)
    print(f"   -> Lendo e fatiando: {nome_arquivo}")
    
    loader = PyPDFLoader(arquivo)
    paginas = loader.load()
    chunks = text_splitter.split_documents(paginas)
    
    for chunk in chunks:
        textos_para_salvar.append(chunk.page_content)
        metadados.append({
            "fonte": nome_arquivo, 
            "pagina": chunk.metadata.get('page', 0)
        })
        ids.append(f"doc_{contador_id}")
        contador_id += 1

print(f"\n   -> Total de {contador_id} blocos de conhecimento gerados!")

# INJEÇÃO NO CHROMADB (EMBEDDINGS)
print("\n3. Calculando Embeddings Vetoriais e salvando no ChromaDB...")
print("   (Isso pode levar alguns minutos dependendo do tamanho dos PDFs. Tenha paciência!)")

colecao.upsert(
    documents=textos_para_salvar,
    metadatas=metadados,
    ids=ids
)

print("\n[SUCESSO EXTREMO] RAG Base alimentada!")
print(f"A sua base de conhecimento agora possui {colecao.count()} vetores de diretrizes médicas.")
print("Ela está pronta para curar as alucinações dos seus modelos de linguagem.")
print("-" * 50)