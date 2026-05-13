import pandas as pd
from llama_cpp import Llama
import re
import gc
import pdfplumber
import os

# LISTA DE QUESTÕES ALVO (SERGIO SANTANA)
ids_alvo = [
    "77.1", "78.1", "79.1", "80.1", "81.1", "82.1", "83.1", "84.1", "85.1", "86.1", 
    "87.1", "89.1", "90.1", "91.1", "92.1", "93.1", "94.1", "95.1", "97.1", "98.1", 
    "99.1", "100.1", "101.1", "103.1", "104.1", "105.1", "106.1"
]

def extrair_dados_pdf(pdf_path):
    print(f"Extraindo texto do PDF: {pdf_path}...")
    questoes_encontradas = []
    texto_completo = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            texto_completo += pagina.extract_text() + "\n"

    # Regex para capturar a questão e o bloco de texto até a próxima questão ou ID
    
    for id_q in ids_alvo:
        pattern = rf"Question {id_q}.*?(?=Question \d+\.1|Answer:|$)"
        match = re.search(pattern, texto_completo, re.DOTALL | re.IGNORECASE)
        
        if match:
            bloco = match.group(0)
            gabarito_match = re.search(r"Answer:\s*([A-E])", bloco, re.IGNORECASE)
            gabarito = gabarito_match.group(1) if gabarito_match else "N/D"
            
            questoes_encontradas.append({
                "ID": id_q,
                "Question": bloco.strip(),
                "Answer": gabarito.upper()
            })
    
    return pd.DataFrame(questoes_encontradas)

# PROCESSAMENTO
caminho_pdf = "seu_arquivo_questoes.pdf" 

if os.path.exists(caminho_pdf):
    df_sergio = extrair_dados_pdf(caminho_pdf)
else:
    print(f"Erro: PDF {caminho_pdf} não encontrado. Usando DataFrame vazio para teste.")
    df_sergio = pd.DataFrame(columns=["ID", "Question", "Answer"])

print(f"Total de questões carregadas: {len(df_sergio)}")

# CONFIGURAÇÃO DOS MODELOS 8GB VRAM
modelos_gguf = [
    ("Llama-3", "./modelos/llama3.gguf"), 
    ("Mistral", "./modelos/mistral.gguf"), 
    ("Phi-3", "./modelos/phi3.gguf")
]

for nome_modelo, caminho in modelos_gguf:
    if not os.path.exists(caminho):
        print(f"Modelo {nome_modelo} não encontrado em {caminho}. Pulando...")
        continue
        
    print(f"\n--- Alocando {nome_modelo} na GPU (8GB VRAM) ---")
    # n_gpu_layers=-1 tudo na GPU. 
    # n_ctx=2048 USMLE e economiza memória.
    llm = Llama(model_path=caminho, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    
    respostas_modelo = []
    
    for idx, row in df_sergio.iterrows():
        prompt = f"""You are a medical specialist. Solve this USMLE Step 3 question.
Reply ONLY with the letter of the correct option (A, B, C, D, or E).

Question:
{row['Question']}

Letter:"""

        try:
            output = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.1
            )
            
            content = output["choices"][0]["message"]["content"].strip().upper()
            match_letra = re.search(r'([A-E])', content)
            letra_final = match_letra.group(1) if match_letra else "N/A"
            respostas_modelo.append(letra_final)
            print(f"Questão {row['ID']}: {letra_final}")
            
        except Exception as e:
            respostas_modelo.append("Erro")
            print(f"Erro na questão {row['ID']}: {e}")
            
    df_sergio[f'Resposta_{nome_modelo}'] = respostas_modelo
    
    # LIMPEZA CRÍTICA DE MEMÓRIA PARA 8GB VRAM
    del llm
    gc.collect()
    # Pequena pausa para o SO liberar a VRAM
    import time
    time.sleep(2)

# CÁLCULOS FINAIS
print("\n--- Relatório de Performance Sergio Santana ---")
for nome_modelo, _ in modelos_gguf:
    col = f'Resposta_{nome_modelo}'
    if col in df_sergio.columns:
        acertos = (df_sergio[col] == df_sergio['Answer']).sum()
        total = len(df_sergio)
        print(f"Acurácia {nome_modelo}: {(acertos/total)*100:.2f}%")

# Concordância
def checar_concordancia(row):
    resps = [str(row.get(f'Resposta_{m[0]}', '')) for m in modelos_gguf]
    if len(set(resps)) == 1: return "Unânime"
    if len(set(resps)) == len(resps): return "Divergente"
    return "Maioria"

df_sergio['Concordância'] = df_sergio.apply(checar_concordancia, axis=1)

# SALVAMENTO
df_sergio.to_excel('M2_FINAL_Sergio_Santana.xlsx', index=False)
print("\nConcluído! Resultados salvos em 'M2_FINAL_Sergio_Santana.xlsx'.")