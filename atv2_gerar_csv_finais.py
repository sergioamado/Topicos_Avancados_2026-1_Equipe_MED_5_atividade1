import pandas as pd

# Modelos que você utilizou na Atividade 1
modelos = ['Llama-3', 'Mistral', 'Phi-3']

print("--- TRANSFORMANDO DATASET M2 (MCQ - MÚLTIPLA ESCOLHA) ---")
try:
    df_m2 = pd.read_excel("M2_FINAL_Sergio.xlsx")
    
    for modelo in modelos:
        df_mcq = pd.DataFrame()
        
        # question
        df_mcq['question'] = df_m2.get('Question', df_m2.get('Pergunta'))
        
        # prediction (com o nome do modelo, ex: llama-3_prediction)
        col_pred = f"{modelo.lower()}_prediction"
        df_mcq[col_pred] = df_m2[f'Resposta_{modelo}']
        
        # correct (gabarito ouro)
        df_mcq['correct'] = df_m2.get('Answer', df_m2.get('Gabarito'))
        
        # score (Calcula automaticamente: 1 se a IA acertou, 0 se a IA errou)
        # Limpamos espaços e deixamos maiúsculo para garantir a comparação
        pred_limpa = df_mcq[col_pred].astype(str).str.strip().str.upper()
        corr_limpa = df_mcq['correct'].astype(str).str.strip().str.upper()
        df_mcq['score'] = (pred_limpa == corr_limpa).astype(int)
        
        nome_arquivo_m2 = f"MCQ_{modelo}.csv"
        df_mcq.to_csv(nome_arquivo_m2, index=False)
        print(f"✅ Salvo: {nome_arquivo_m2}")

except FileNotFoundError:
    print("Arquivo 'M2_FINAL_Sergio.xlsx' não encontrado. Verifique o nome.")


print("\n--- TRANSFORMANDO DATASET M1 (OPEN - QUESTÕES ABERTAS) ---")
try:
    df_m1 = pd.read_excel("M1_FINAL_Sergio.xlsx")
    
    for modelo in modelos:
        df_open = pd.DataFrame()
        
        # id
        # Se você já tiver uma coluna de ID, ele pega. Se não, cria números de 1 até o fim.
        if 'ID_Questao' in df_m1.columns:
            df_open['id'] = df_m1['ID_Questao']
        else:
            df_open['id'] = range(1, len(df_m1) + 1)
            
        # question
        df_open['question'] = df_m1.get('Pergunta', df_m1.get('Question'))
        
        # answer (A resposta que a IA gerou)
        df_open['answer'] = df_m1[f'Resposta_{modelo}']
        
        # must_have_score
        # Tenta procurar alguma nota que você já tenha (Juiz, BERTScore, ou TokenF1).
        # Se você tiver uma coluna com outro nome, basta alterar aqui embaixo:
        if f'Nota_Juiz_{modelo}' in df_m1.columns:
            df_open['must_have_score'] = df_m1[f'Nota_Juiz_{modelo}']
        elif f'BERTScore_{modelo}' in df_m1.columns:
            df_open['must_have_score'] = df_m1[f'BERTScore_{modelo}']
        else:
            # Deixa vazio para não quebrar a formatação caso a coluna não exista
            df_open['must_have_score'] = "" 
            
        nome_arquivo_m1 = f"Open_{modelo}.csv"
        df_open.to_csv(nome_arquivo_m1, index=False)
        print(f"✅ Salvo: {nome_arquivo_m1}")

except FileNotFoundError:
    print("Arquivo 'M1_FINAL_Sergio.xlsx' não encontrado. Verifique o nome.")

print("\n🎉 Transformação concluída com sucesso!")