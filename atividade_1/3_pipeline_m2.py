import pandas as pd
from llama_cpp import Llama
import re
import gc
import os
import time

# --- CONFIGURAÇÕES DE PODER MÁXIMO ---
N_CTX = 4096         # Aumentado para suportar casos médicos longos + raciocínio
N_BATCH = 1024       # Aumentado para processar o prompt mais rápido na 3060
GPU_LAYERS = -1      # Tudo na GPU

def extrair_letra(texto):
    # Procura por "Resposta: A" ou "Resultado: A" ou apenas a letra isolada no final
    match = re.search(r'(?:RESPOSTA|ANSWER|RESULTADO|FINAL):\s*([A-E])', texto, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Se não achar o padrão, tenta achar qualquer letra A-E isolada
    match = re.search(r'\b[A-E]\b', texto)
    return match.group(0).upper() if match else "N/A"

print("Lendo Dataset M2...")
df_m2 = pd.read_excel('dataset_m2.xlsx') 
suas_questoes = df_m2.copy() 

# Lista de modelos (Sugestão: use versões Q5_K_M ou Q6_K para máxima qualidade na 3060)
modelos_gguf = [
    ("Llama-3", "./modelos/llama3.gguf"), 
    ("Mistral", "./modelos/mistral.gguf"), 
    ("Phi-3", "./modelos/phi3.gguf")
]

for nome_modelo, caminho in modelos_gguf:
    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        continue

    print(f"\n" + "="*50)
    print(f"CARREGANDO MODELO: {nome_modelo}")
    print("="*50)
    
    llm = Llama(
        model_path=caminho,
        n_gpu_layers=GPU_LAYERS,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        f16_kv=True,
        flash_attn=True, # Crucial para performance na série 3000
        verbose=False
    )
    
    respostas_finais = []
    raciocinios = []
    
    start_time = time.time()
    
    for idx, row in suas_questoes.iterrows():
        try:
            # PROMPT AVANÇADO (Chain of Thought)
            # Pedimos para o modelo pensar brevemente, isso aumenta a acurácia médica.
            prompt_sistema = (
                "You are a highly skilled medical specialist. "
                "Analyze the following clinical case, provide a very brief rationale, "
                "and then conclude with the correct option letter."
            )
            
            prompt_usuario = (
                f"Question: {row['Question']}\n\n"
                "Format your response as:\n"
                "Rationale: [Your brief reasoning]\n"
                "Answer: [Letter]"
            )

            resposta = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt_usuario}
                ],
                max_tokens=150, # Espaço para o raciocínio
                temperature=0.1  # Baixa temperatura para manter consistência
            )
            
            conteudo = resposta["choices"][0]["message"]["content"]
            letra = extrair_letra(conteudo)
            
            respostas_finais.append(letra)
            raciocinios.append(conteudo) # Guardamos o raciocínio para análise posterior
            
            if idx % 5 == 0:
                print(f"[{nome_modelo}] Processado {idx}/{len(suas_questoes)}... Última: {letra}")
                
        except Exception as e:
            print(f"Erro no índice {idx}: {e}")
            respostas_finais.append("Erro")
            raciocinios.append("Erro")
    
    suas_questoes[f'Resposta_{nome_modelo}'] = respostas_finais
    suas_questoes[f'Raciocinio_{nome_modelo}'] = raciocinios
    
    # Limpeza total da VRAM para o próximo modelo
    del llm
    gc.collect()
    time.sleep(2) # Pausa para o driver da GPU respirar

# --- MÉTRICAS E FINALIZAÇÃO ---
print("\n" + "X"*50)
print("PROCESSAMENTO CONCLUÍDO. GERANDO RESULTADOS...")

if 'Answer' in suas_questoes.columns:
    gabarito = suas_questoes['Answer'].astype(str).str.strip().str.upper()
    for nome, _ in modelos_gguf:
        if f'Resposta_{nome}' in suas_questoes.columns:
            preds = suas_questoes[f'Resposta_{nome}'].astype(str).str.strip().str.upper()
            acertos = (preds == gabarito).sum()
            total = len(gabarito)
            print(f"Acurácia Final {nome}: {acertos}/{total} ({(acertos/total)*100:.2f}%)")

# Concordância entre os 3 principais
def verificar_concordancia(row):
    try:
        r1 = str(row.get(f'Resposta_{modelos_gguf[0][0]}'))
        r2 = str(row.get(f'Resposta_{modelos_gguf[1][0]}'))
        r3 = str(row.get(f'Resposta_{modelos_gguf[2][0]}'))
        if r1 == r2 == r3: return "Unânime"
        if r1 == r2 or r1 == r3 or r2 == r3: return "Maioria"
        return "Divergente"
    except:
        return "Erro"

suas_questoes['Consenso'] = suas_questoes.apply(verificar_concordancia, axis=1)

# Salva com os raciocínios para você poder conferir por que o modelo escolheu a letra
suas_questoes.to_excel('M2_RESULTADO_MAXIMO.xlsx', index=False)
print("\nArquivo salvo: M2_RESULTADO_MAXIMO.xlsx")