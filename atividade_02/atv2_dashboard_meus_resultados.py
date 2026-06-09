import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Meus Resultados - Sergio", layout="wide")

st.title("👨‍💻 Análise Individual: Os Meus Modelos (Equipa 5)")
st.markdown("Análise focada no desempenho do **Llama-3, Mistral e Phi-3** nas Questões Abertas (M1) e Múltipla Escolha (M2).")

# Modelos do Sergio
meus_modelos = ['Llama-3', 'Mistral', 'Phi-3']

#  CARREGAR DADOS M1 (Questões Abertas / Juiz) 
@st.cache_data
def carregar_m1():
    try:
        df = pd.read_csv("avaliacoes_consolidadas_equipe5.csv")
        df['nota_do_juiz'] = pd.to_numeric(df['nota_do_juiz'], errors='coerce')
        # Filtra apenas para os modelos do Sergio e para o dataset de questões abertas
        df_m1 = df[df['modelo_avaliado'].str.contains('|'.join(meus_modelos), case=False, na=False)]
        df_m1 = df_m1[df_m1['dataset'].str.contains('Abertas|M1|K-QA', case=False, na=False)]
        return df_m1
    except Exception as e:
        return pd.DataFrame()

#  CARREGAR DADOS M2 (Múltipla Escolha / Exatidão) 
@st.cache_data
def carregar_m2():
    dfs = []
    for modelo in meus_modelos:
        arquivo = f"MCQ_{modelo}.csv"
        if os.path.exists(arquivo):
            df_temp = pd.read_csv(arquivo)
            df_temp['modelo'] = modelo
            dfs.append(df_temp)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

df_m1 = carregar_m1()
df_m2 = carregar_m2()

aba1, aba2 = st.tabs(["📝 M1: Questões Abertas (Avaliação do Juiz)", "🎯 M2: Múltipla Escolha (Taxa de Acerto)"])

# QUESTÕES ABERTAS (M1)
with aba1:
    st.header("Desempenho Raciocínio Clínico (M1)")
    if df_m1.empty:
        st.warning("Não foram encontrados dados das questões abertas para os seus modelos.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Nota Média por Modelo")
            df_media_m1 = df_m1.groupby('modelo_avaliado')['nota_do_juiz'].mean().reset_index()
            fig_bar_m1 = px.bar(
                df_media_m1, x='modelo_avaliado', y='nota_do_juiz', 
                color='modelo_avaliado', text_auto='.2f',
                labels={'modelo_avaliado': 'Meus Modelos', 'nota_do_juiz': 'Nota (1 a 5)'}
            )
            st.plotly_chart(fig_bar_m1, use_container_width=True)
            
        with col2:
            st.subheader("Distribuição de Notas (Qualidade)")
            def classificar(nota):
                if nota >= 4: return 'Bom/Excelente (4-5)'
                elif nota == 3: return 'Regular (3)'
                else: return 'Ruim/Crítico (1-2)'
            
            df_m1['qualidade'] = df_m1['nota_do_juiz'].apply(classificar)
            df_qualidade = df_m1.groupby(['modelo_avaliado', 'qualidade']).size().reset_index(name='qtd')
            fig_stack = px.bar(
                df_qualidade, x="modelo_avaliado", y="qtd", color="qualidade",
                color_discrete_map={'Bom/Excelente (4-5)': '#28a745', 'Regular (3)': '#ffc107', 'Ruim/Crítico (1-2)': '#dc3545'},
                barmode='group'
            )
            st.plotly_chart(fig_stack, use_container_width=True)

# MÚLTIPLA ESCOLHA (M2)
with aba2:
    st.header("Exatidão em Exames Médicos (M2)")
    st.markdown("*A métrica aqui é exata: 1 para acerto e 0 para erro (Acurácia).*")
    
    if df_m2.empty:
        st.warning("Os arquivos MCQ_...csv não foram encontrados na pasta.")
    else:
        # Calcula a taxa de acerto (acurácia) em porcentagem
        df_acc = df_m2.groupby('modelo')['score'].mean().reset_index()
        df_acc['score'] = df_acc['score'] * 100 # Transforma em %
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(df_acc.rename(columns={'modelo': 'Modelo', 'score': 'Taxa de Acerto (%)'}).style.format({'Taxa de Acerto (%)': '{:.1f}%'}))
            
        with col2:
            fig_acc = px.bar(
                df_acc, x='modelo', y='score', color='modelo',
                text_auto='.1f', title="Taxa de Acerto (Acurácia %)",
                labels={'modelo': 'Modelo', 'score': 'Acertos (%)'}
            )
            fig_acc.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig_acc.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_acc, use_container_width=True)
            
        st.divider()
        st.markdown("### Auditoria de Erros em Múltipla Escolha")
        modelo_filtro = st.selectbox("Selecione um modelo para ver onde ele errou:", df_m2['modelo'].unique())
        erros_modelo = df_m2[(df_m2['modelo'] == modelo_filtro) & (df_m2['score'] == 0)]
        
        st.error(f"O modelo **{modelo_filtro}** errou {len(erros_modelo)} questões.")
        st.dataframe(erros_modelo[['question', f'{modelo_filtro.lower()}_prediction', 'correct']])