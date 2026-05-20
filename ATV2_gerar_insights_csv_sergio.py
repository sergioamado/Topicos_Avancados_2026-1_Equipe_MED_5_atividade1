import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. CONFIGURAÇÃO ÚNICA DA PÁGINA (Deve ser a primeira linha)
st.set_page_config(page_title="Portal de Auditoria IA - Equipa 5", layout="wide", initial_sidebar_state="expanded")

# --- CSS CUSTOMIZADO PARA ESTÉTICA PROFISSIONAL ---
st.markdown("""
    <style>
    .metric-box { background-color: #f0f2f6; padding: 15px; border-radius: 8px; text-align: center; }
    .metric-title { font-size: 14px; color: #555; }
    .metric-value { font-size: 24px; font-weight: bold; color: #0f52ba; }
    </style>
""", unsafe_allow_html=True)

# --- CARREGAMENTO DE DADOS (CACHE UNIFICADO) ---
@st.cache_data
def load_dados_m1():
    try:
        df = pd.read_csv("avaliacoes_consolidadas_equipe5.csv")
        df['nota_do_juiz'] = pd.to_numeric(df['nota_do_juiz'], errors='coerce')
        
        def categorizar_modelo(nome):
            nome_upper = str(nome).upper()
            if any(x in nome_upper for x in ['GEMINI', 'GPT', 'GROK', 'PERPLEXITY', 'CLAUDE']):
                return 'Gigantes (Nuvem)'
            else:
                return 'Locais (Até 8B)'
                
        df['categoria_modelo'] = df['modelo_avaliado'].apply(categorizar_modelo)
        
        def classificar_desempenho(nota):
            if nota >= 4: return 'Muito Bom (4-5)'
            elif nota == 3: return 'Regular (3)'
            else: return 'Ruim / Crítico (1-2)'
            
        df['qualidade'] = df['nota_do_juiz'].apply(classificar_desempenho)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_dados_m2(modelos_sergio):
    dfs = []
    for modelo in modelos_sergio:
        arquivo = f"MCQ_{modelo}.csv"
        if os.path.exists(arquivo):
            df_temp = pd.read_csv(arquivo)
            df_temp['modelo'] = modelo
            dfs.append(df_temp)
    if dfs: return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

df_m1 = load_dados_m1()
meus_modelos = ['Llama-3', 'Mistral', 'Phi-3']
df_m2 = load_dados_m2(meus_modelos)

if df_m1.empty:
    st.error("⚠️ Ficheiro 'avaliacoes_consolidadas_equipe5.csv' não encontrado. Pare a aplicação e gere os dados.")
    st.stop()

# --- MENU DE NAVEGAÇÃO LATERAL ---
st.sidebar.title("🩺 Navegação")
menu = st.sidebar.radio("Selecione o Painel:", [
    "🌍 Visão Global (Equipa)", 
    "⚖️ Benchmarking (Gap vs Nuvem)", 
    "👨‍💻 Análise Individual (Sérgio)"
])

st.sidebar.divider()

# =====================================================================
# PAINEL 1: VISÃO GLOBAL (A antiga aplicação 1)
# =====================================================================
if menu == "🌍 Visão Global (Equipa)":
    st.title("🌍 Visão Global da Equipa: LLM-as-a-Judge")
    st.markdown("Análise de segurança e acurácia clínica sob a avaliação do Llama-3.3-70B.")

    # Filtros
    modelos_disponiveis = df_m1['modelo_avaliado'].unique()
    modelos_selecionados = st.sidebar.multiselect("Comparar Modelos:", modelos_disponiveis, default=modelos_disponiveis)
    df_filtrado = df_m1[df_m1['modelo_avaliado'].isin(modelos_selecionados)]

    aba1, aba2, aba3 = st.tabs(["📊 Proporção de Qualidade", "🚨 Risco Clínico", "🧩 Correlação de Spearman"])

    with aba1:
        st.subheader("Proporção de Respostas por Qualidade")
        df_count = df_filtrado.groupby(['modelo_avaliado', 'qualidade']).size().reset_index(name='contagem')
        fig_stack = px.bar(
            df_count, x="modelo_avaliado", y="contagem", color="qualidade",
            color_discrete_map={'Muito Bom (4-5)': '#28a745', 'Regular (3)': '#ffc107', 'Ruim / Crítico (1-2)': '#dc3545'},
            barmode='relative', labels={'modelo_avaliado': 'Modelo', 'contagem': 'Volume'}
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    with aba2:
        st.subheader("🚨 Eventos Adversos (Nota 1)")
        df_criticos = df_filtrado[df_filtrado['nota_do_juiz'] == 1].copy()
        if df_criticos.empty:
            st.success("Nenhuma falha crítica encontrada.")
        else:
            st.warning(f"Foram identificadas {len(df_criticos)} condutas médicas de altíssimo risco.")
            for index, row in df_criticos.head(5).iterrows():
                with st.expander(f"⚠️ Erro Crítico: {row['modelo_avaliado']}"):
                    st.write(f"**Pergunta:** {row['pergunta']}")
                    st.write(f"**IA:** {row['resposta_da_ia']}")
                    st.write(f"**Juiz:** {row['justificativa_do_juiz']}")

    with aba3:
        st.subheader("Matriz de Proximidade (Spearman)")
        df_pivot = df_filtrado.pivot_table(index='id_pergunta', columns='modelo_avaliado', values='nota_do_juiz')
        fig_corr = px.imshow(df_pivot.corr(method='spearman'), text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

# =====================================================================
# PAINEL 2: BENCHMARKING (O Novo Requisito de Comparação de Desempenho)
# =====================================================================
elif menu == "⚖️ Benchmarking (Gap vs Nuvem)":
    st.title("⚖️ Benchmarking: Locais vs Gigantes da Nuvem")
    st.markdown("Comparativo de desempenho paramétrico. Quanto cada modelo é melhor ou pior que o padrão de mercado?")

    # Calcula a Linha de Base (Média dos Gigantes)
    df_gigantes = df_m1[df_m1['categoria_modelo'] == 'Gigantes (Nuvem)']
    baseline_score = df_gigantes['nota_do_juiz'].mean() if not df_gigantes.empty else 0

    if baseline_score == 0:
        st.warning("Não há modelos 'Gigantes' suficientes no dataset para criar a linha de base.")
    else:
        # Calcula a média de TODOS os modelos
        df_medias = df_m1.groupby('modelo_avaliado')['nota_do_juiz'].mean().reset_index()
        
        # Calcula a diferença percentual de cada modelo em relação à linha de base
        df_medias['gap_pct'] = ((df_medias['nota_do_juiz'] - baseline_score) / baseline_score) * 100
        df_medias['cor'] = df_medias['gap_pct'].apply(lambda x: 'Melhor que a Nuvem' if x > 0 else 'Abaixo da Nuvem')

        # Cria as colunas superiores
        c1, c2, c3 = st.columns(3)
        c1.metric("Linha de Base (Média Nuvem)", f"{baseline_score:.2f} / 5.00")
        c2.metric("Melhor Modelo Avaliado", df_medias.loc[df_medias['nota_do_juiz'].idxmax()]['modelo_avaliado'])
        c3.metric("Pior Modelo Avaliado", df_medias.loc[df_medias['nota_do_juiz'].idxmin()]['modelo_avaliado'])

        st.divider()

        # O GRÁFICO DE GAP (Diverging Bar Chart)
        st.subheader("Desvio Percentual de Desempenho (%)")
        st.markdown(f"*Referência 0% = Média dos Gigantes da Nuvem ({baseline_score:.2f})*")
        
        # Ordenar para ficar bonito
        df_medias = df_medias.sort_values('gap_pct')
        
        fig_gap = px.bar(
            df_medias, x='gap_pct', y='modelo_avaliado', orientation='h', color='cor',
            color_discrete_map={'Melhor que a Nuvem': '#28a745', 'Abaixo da Nuvem': '#dc3545'},
            text_auto='.1f', labels={'gap_pct': 'Diferença Percentual (%)', 'modelo_avaliado': ''}
        )
        fig_gap.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
        fig_gap.update_traces(texttemplate='%{x:+.1f}%', textposition='outside')
        st.plotly_chart(fig_gap, use_container_width=True)

        # Gráfico Boxplot para mostrar consistência
        st.subheader("Dispersão e Consistência (Boxplot)")
        fig_box = px.box(df_m1, x='modelo_avaliado', y='nota_do_juiz', color='categoria_modelo', points="all")
        st.plotly_chart(fig_box, use_container_width=True)

# =====================================================================
# PAINEL 3: ANÁLISE INDIVIDUAL (A antiga aplicação 2)
# =====================================================================
elif menu == "👨‍💻 Análise Individual (Sérgio)":
    st.title("👨‍💻 Análise Individual: Os Meus Modelos")
    st.markdown(f"Foco exclusivo nos modelos: **{', '.join(meus_modelos)}**")

    # Filtra apenas os modelos do Sergio
    df_m1_sergio = df_m1[df_m1['modelo_avaliado'].str.contains('|'.join(meus_modelos), case=False, na=False)]

    t1, t2 = st.tabs(["📝 M1: Abertas (Juiz)", "🎯 M2: MCQ (Acurácia)"])

    with t1:
        st.subheader("Desempenho no Raciocínio Clínico (Nota 1-5)")
        c1, c2 = st.columns(2)
        with c1:
            df_media = df_m1_sergio.groupby('modelo_avaliado')['nota_do_juiz'].mean().reset_index()
            fig_bar = px.bar(df_media, x='modelo_avaliado', y='nota_do_juiz', color='modelo_avaliado', text_auto='.2f')
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            df_qualidade = df_m1_sergio.groupby(['modelo_avaliado', 'qualidade']).size().reset_index(name='qtd')
            fig_stack = px.bar(df_qualidade, x="modelo_avaliado", y="qtd", color="qualidade", barmode='group')
            st.plotly_chart(fig_stack, use_container_width=True)

    with t2:
        st.subheader("Exatidão em Múltipla Escolha (USMLE)")
        if df_m2.empty:
            st.warning("Ficheiros MCQ não encontrados para carregar M2.")
        else:
            df_acc = df_m2.groupby('modelo')['score'].mean().reset_index()
            df_acc['score'] *= 100
            fig_acc = px.bar(df_acc, x='modelo', y='score', color='modelo', text_auto='.1f', title="Taxa de Acerto (%)")
            fig_acc.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig_acc.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_acc, use_container_width=True)