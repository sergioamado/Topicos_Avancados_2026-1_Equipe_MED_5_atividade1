import pandas as pd
from llama_cpp import Llama
import re
import gc

print("Lendo Dataset M2 local...")
df_m2 = pd.read_excel('dataset_m2.xlsx') 

suas_questoes = df_m2.copy() 

modelos_gguf = [
    ("Llama-3", "./modelos/llama3.gguf"), 
    ("Mistral", "./modelos/mistral.gguf"), 
    ("Phi-3", "./modelos/phi3.gguf")
]

print("\nIniciando extração de alternativas...")
for nome_modelo, caminho in modelos_gguf:
    print(f"Processando {nome_modelo}...")
    llm = Llama(model_path=caminho, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    respostas = []
    
    for idx, row in suas_questoes.iterrows():
        try:
            resposta = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are taking a multiple-choice medical test. Reply ONLY with the single letter of the correct option (A, B, C, D, or E). Do not explain."},
                    {"role": "user", "content": str(row['Question'])}
                ], max_tokens=5, temperature=0.1
            )
            letra = re.search(r'\b[A-E]\b', resposta["choices"][0]["message"]["content"])
            respostas.append(letra.group() if letra else "N/A")
        except: 
            respostas.append("Erro")
        
    suas_questoes[f'Resposta_{nome_modelo}'] = respostas
    del llm
    gc.collect()

print("\nCalculando Acurácia e Concordância...")
gabaritos = suas_questoes['Answer'].astype(str).str.strip().str.upper().tolist()

for nome, _ in modelos_gguf:
    resp = suas_questoes[f'Resposta_{nome}'].astype(str).str.strip().str.upper().tolist()
    acertos = sum(1 for r, g in zip(resp, gabaritos) if r == g)
    print(f"Acurácia {nome}: {(acertos / len(gabaritos)) * 100:.2f}%")

# Métrica de Concordância Múltipla Escolha
concordancia = []
for idx, row in suas_questoes.iterrows():
    r_l3 = str(row.get('Resposta_Llama-3')).strip()
    r_m = str(row.get('Resposta_Mistral')).strip()
    r_p3 = str(row.get('Resposta_Phi-3')).strip()
    
    if r_l3 == r_m == r_p3: 
        concordancia.append("Unânime")
    else: 
        concordancia.append("Divergente")
        
suas_questoes['Concordância_Modelos'] = concordancia

suas_questoes.to_excel('M2_FINAL_Sergio.xlsx', index=False)
print("\nConcluído! Salvo em 'M2_FINAL_Sergio.xlsx'.")