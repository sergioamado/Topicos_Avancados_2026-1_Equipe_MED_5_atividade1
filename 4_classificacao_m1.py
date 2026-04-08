import pandas as pd
from llama_cpp import Llama
from bert_score import score
import matplotlib.pyplot as plt
import difflib
import gc
import torch
import warnings
import re

warnings.filterwarnings('ignore')

# =========================
# CONFIGURAÇÃO DE AMBIENTE
# =========================
INPUT_FILE = "M1_FINAL_Sergio.xlsx"
OUTPUT_FILE = "M1_RESULTADO_COMPLETO.xlsx"

MODELS = [
    ("LLaMA", "modelos/llama3.gguf"),
    ("Mistral", "modelos/mistral.gguf"),
    ("Phi3", "modelos/phi3.gguf")
]

# =========================
# FUNÇÕES DE APOIO E MÉTRICAS
# =========================

def normalize_class(text):
    text = str(text).lower().strip()
    if "expert" in text: return "Expert"
    if "especialista" in text: return "Especialista"
    if "generalista" in text: return "Generalista"
    if "triagem" in text: return "Triagem"
    return "Indefinido"

def majority_vote(votes):
    count = {}
    for v in votes:
        count[v] = count.get(v, 0) + 1
    sorted_votes = sorted(count.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_votes) == 3 and sorted_votes[0][1] == 1:
        return "Sem_Consenso"
    return sorted_votes[0][0]

def limpar_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clean_tokens(text):
    """Remove pontuação para um cálculo de F1-Token rigoroso e livre de ruídos."""
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return set(text.split())

def calc_f1(g, r):
    g_tok, r_tok = clean_tokens(g), clean_tokens(r)
    if not g_tok or not r_tok: return 0.0
    intsec = g_tok.intersection(r_tok)
    p, rec = len(intsec) / len(g_tok), len(intsec) / len(r_tok)
    return 0.0 if p + rec == 0 else 2 * (p * rec) / (p + rec)

def calc_desvio(t1, t2):
    sim = difflib.SequenceMatcher(None, str(t1).lower(), str(t2).lower()).ratio()
    return round((1 - sim) * 100, 2)

# =========================
# MOTOR PRINCIPAL
# =========================

def main():
    print("1. Carregando Dataset e Estruturas...")
    df = pd.read_excel(INPUT_FILE)
    gabaritos = df['Gabarito'].tolist()
    all_results = {}

    # =========================
    # FASE 1: INFERÊNCIA LLM (Exclusivo GPU RTX 3060)
    # =========================
    print("\n2. Iniciando Inferência com Aceleração de Hardware...")
    
    for nome_modelo, caminho in MODELS:
        print(f"\n=== Alocando {nome_modelo} na VRAM ===")
        all_results[nome_modelo] = {"answers": [], "classes": []}
        
        try:
            llm = Llama(model_path=caminho, n_gpu_layers=-1, n_ctx=2048, verbose=False)
        except Exception as e:
            print(f"Falha de hardware ao carregar {nome_modelo}: {e}")
            continue

        for i, row in df.iterrows():
            question = str(row["Pergunta"])
            gold = str(row["Gabarito"])
            
            # Execução estruturada via ChatML/Instruct
            try:
                resp_out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are an expert medical professional. Answer the medical question clearly, technically, and based on evidence."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=400, temperature=0.1
                )
                resposta_gerada = resp_out["choices"][0]["message"]["content"].strip()
            except Exception as e:
                resposta_gerada = f"Erro de processamento: {e}"

            # Classificação rígida
            try:
                class_out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "Classify the following medical question into exactly one of these four categories: Triagem, Generalista, Especialista, or Expert. Respond with ONLY the category name."},
                        {"role": "user", "content": f"Question:\n{question}\n\nExpected Answer:\n{gold}"}
                    ],
                    max_tokens=10, temperature=0.0
                )
                classe_gerada = normalize_class(class_out["choices"][0]["message"]["content"])
            except:
                classe_gerada = "Erro_Classificacao"

            all_results[nome_modelo]["answers"].append(resposta_gerada)
            all_results[nome_modelo]["classes"].append(classe_gerada)
            
            print(f"[{nome_modelo}] Bloco concluído: {i+1}/{len(df)}", end="\r")
            
        del llm
        limpar_vram()

    # =========================
    # FASE 2: CÁLCULO VETORIZADO BERTScore
    # =========================
    print("\n\n3. Computando Malha Semântica (BERTScore)...")
    
    for nome_modelo, data in all_results.items():
        respostas = data["answers"]
        P, R, F1_bert = score(respostas, gabaritos, lang="en", verbose=False, rescale_with_baseline=True, device="cuda" if torch.cuda.is_available() else "cpu")
        all_results[nome_modelo]["bert_scores"] = F1_bert.numpy().tolist()
        
    limpar_vram()

    # =========================
    # FASE 3: CONSOLIDAÇÃO E MÉTRICAS LÉXICAS AVANÇADAS
    # =========================
    print("\n4. Calculando F1-Token, Divergência e Consolidando Arquitetura de Dados...")
    
    final_votes = []
    for i in range(len(df)):
        votes = [all_results[m[0]]["classes"][i] for m in MODELS if m[0] in all_results]
        final_votes.append(majority_vote(votes))

    df_out = pd.DataFrame({
        "Pergunta": df["Pergunta"],
        "Gabarito": df["Gabarito"],
        "Voto_Final": final_votes
    })

    # Injeção de Respostas, Classes e F1
    for m, _ in MODELS:
        if m in all_results:
            respostas_modelo = all_results[m]["answers"]
            df_out[f"Answer_{m}"] = respostas_modelo
            df_out[f"Class_{m}"] = all_results[m]["classes"]
            df_out[f"BERTScore_{m}"] = all_results[m]["bert_scores"]
            # F1-Token processado via list comprehension para evitar gargalo iterativo
            df_out[f"TokenF1_{m}"] = [calc_f1(g, r) for g, r in zip(gabaritos, respostas_modelo)]

    # Análise de Desvio Inter-Modelo (Se todos os 3 modelos foram processados com sucesso)
    if len([m for m in MODELS if m[0] in all_results]) == 3:
        ans_llama = all_results["LLaMA"]["answers"]
        ans_mistral = all_results["Mistral"]["answers"]
        ans_phi = all_results["Phi3"]["answers"]

        df_out['Desvio_Llama_Mistral(%)'] = [calc_desvio(l, m) for l, m in zip(ans_llama, ans_mistral)]
        df_out['Desvio_Llama_Phi3(%)'] = [calc_desvio(l, p) for l, p in zip(ans_llama, ans_phi)]
        df_out['Desvio_Mistral_Phi3(%)'] = [calc_desvio(m, p) for m, p in zip(ans_mistral, ans_phi)]

    df_out.to_excel(OUTPUT_FILE, index=False)

    # =========================
    # FASE 4: DIAGNÓSTICO ESTÁTICO DE DADOS
    # =========================
    print("\n5. Extração de Componentes Visuais...")
    
    model_scores = {m[0]: df_out[f"BERTScore_{m[0]}"].mean() for m in MODELS if m[0] in all_results}
    ranking = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n=== RANKING DE PRECISÃO SINTÉTICA (BERTScore) ===")
    for i, (model, score_val) in enumerate(ranking, 1):
        f1_medio = df_out[f"TokenF1_{model}"].mean()
        print(f"{i}º {model} - BERT: {score_val:.4f} | F1-Token: {f1_medio:.4f}")

    score_cols = [f"BERTScore_{m[0]}" for m in MODELS if m[0] in all_results]
    grouped = df_out.groupby("Voto_Final")[score_cols].mean()

    # Operações de plotagem
    plt.figure()
    plt.bar(model_scores.keys(), model_scores.values(), color=['#2c3e50', '#e74c3c', '#27ae60'])
    plt.title("Avaliação de Embedding por Modelo")
    plt.ylabel("F1 Score Rescalonado")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("ranking_modelos.png")
    plt.close()

    grouped.plot(kind="bar", figsize=(10, 6), colormap='viridis')
    plt.title("Desempenho Vetorial Relativo por Escala de Complexidade")
    plt.ylabel("Média F1 Score")
    plt.xticks(rotation=0)
    plt.legend(title="Modelos")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("score_por_classe.png")
    plt.close()

    print(f"\nRotina de testes finalizada. Saída pragmática confirmada em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()