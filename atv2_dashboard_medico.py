import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Auditoria Clínica IA - Equipa 5", layout="wide", initial_sidebar_state="expanded")

st.title("🩺 Auditoria Clínica de IA: LLM-as-a-Judge")
st.markdown("Análise de segurança, acurácia e risco ao paciente sob a avaliação do Llama-3.3-70B (Diretrizes AHA/SBC).")

# CARREGAMENTO E PREPARAÇÃO DOS DADOS 
@st.cache_data
def load_data():
    df = pd.read_csv("avaliacoes_consolidadas_equipe5.csv")
    df['nota_do_juiz'] = pd.to_numeric(df['nota_do_juiz'], errors='coerce')
    
    # Classificador de Modelos (Pequenos/Locais vs Grandes/Nuvem)
    def categorizar_modelo(nome):
        nome_upper = str(nome).upper()
        if any(x in nome_upper for x in ['GEMINI', 'GPT', 'GROK', 'PERPLEXITY']):
            return 'Gigantes (Nuvem)'
        else:
            return 'Locais (Até 8B)'
            
    df['categoria_modelo'] = df['modelo_avaliado'].apply(categorizar_modelo)
    
    # Classificador de Qualidade
    def classificar_desempenho(nota):
        if nota >= 4: return 'Muito Bom (4-5)'
        elif nota == 3: return 'Regular (3)'
        else: return 'Muito Ruim / Crítico (1-2)'
        
    df['qualidade'] = df['nota_do_juiz'].apply(classificar_desempenho)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Ficheiro CSV não encontrado. Execute a exportação primeiro.")
    st.stop()

# FILTROS LATERAIS 
st.sidebar.header("🔍 Filtros Clínicos")
dataset_selecionado = st.sidebar.selectbox("Dataset:", df['dataset'].unique())
df_filtrado = df[df['dataset'] == dataset_selecionado]

modelos_disponiveis = df_filtrado['modelo_avaliado'].unique()
modelos_selecionados = st.sidebar.multiselect("Comparar Modelos:", modelos_disponiveis, default=modelos_disponiveis)
df_filtrado = df_filtrado[df_filtrado['modelo_avaliado'].isin(modelos_selecionados)]

# Filtro Textual
termo_busca = st.sidebar.text_input("Filtrar por Doença/Termo (ex: hep, heart, blood):")
if termo_busca:
    df_filtrado = df_filtrado[df_filtrado['pergunta'].str.contains(termo_busca, case=False, na=False)]

st.sidebar.divider()
st.sidebar.markdown("📥 **Exportar Dados Filtrados**")
csv_export = df_filtrado.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label="Descarregar CSV Atual", data=csv_export, file_name="auditoria_filtrada.csv", mime="text/csv")


# CÁLCULO DE GAP TECNOLÓGICO % DIFERENÇA
st.markdown("### 🏆 Painel de Desempenho e Indicadores")
col1, col2, col3, col4 = st.columns(4)

total_avaliacoes = len(df_filtrado)
media_geral = df_filtrado['nota_do_juiz'].mean()

# Médias por categoria
medias_cat = df_filtrado.groupby('categoria_modelo')['nota_do_juiz'].mean()
media_locais = medias_cat.get('Locais (Até 8B)', 0)
media_gigantes = medias_cat.get('Gigantes (Nuvem)', 0)

# Calcula o percentual de diferença
if media_locais > 0 and media_gigantes > 0:
    diff_pct = ((media_gigantes - media_locais) / media_locais) * 100
else:
    diff_pct = 0.0

col1.metric("Avaliações Filtradas", total_avaliacoes)
col2.metric("Média Global (1-5)", f"{media_geral:.2f}")
col3.metric("Média Gigantes (Nuvem)", f"{media_gigantes:.2f}")

# Métrica comparativa
if diff_pct > 0:
    col4.metric("Vantagem da Nuvem", f"{media_gigantes:.2f}", f"+{diff_pct:.1f}% vs Locais", delta_color="normal")
elif diff_pct < 0:
    col4.metric("Desvantagem da Nuvem", f"{media_gigantes:.2f}", f"{diff_pct:.1f}% vs Locais", delta_color="inverse")
else:
    col4.metric("Diferença de Desempenho", "Empate / NA", "0%")

st.divider()

# CRIAÇÃO DAS ABAS
aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Desempenho (Bons x Ruins)", 
    "⚖️ Gigantes vs Locais", 
    "🚨 Risco Clínico & Urgências", 
    "🧩 Matriz de Proximidade"
])


# ABA 1: PROPORÇÃO DE QUALIDADE CLÍNICA
with aba1:
    st.markdown("#### Proporção de Qualidade Clínica")
    df_count = df_filtrado.groupby(['modelo_avaliado', 'qualidade']).size().reset_index(name='contagem')
    fig_stack = px.bar(
        df_count, x="modelo_avaliado", y="contagem", color="qualidade",
        color_discrete_map={'Muito Bom (4-5)': '#28a745', 'Regular (3)': '#ffc107', 'Muito Ruim / Crítico (1-2)': '#dc3545'},
        labels={'modelo_avaliado': 'Modelo', 'contagem': 'Qtd de Respostas'}, barmode='relative'
    )
    st.plotly_chart(fig_stack, use_container_width=True)


# ABA 2: JUIZ 70B (COMPARAÇÃO DETALHADA)
with aba2:
    st.markdown("#### Distribuição Exata das Notas: Gigantes vs Locais")
    st.markdown("Os modelos comerciais de facto justificam o seu custo com um desempenho clinicamente superior?")
    
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        # Gráfico Radar/Polar para médias
        df_media_modelo = df_filtrado.groupby('modelo_avaliado')['nota_do_juiz'].mean().reset_index()
        fig_bar = px.bar(
            df_media_modelo.sort_values('nota_do_juiz', ascending=False), 
            x='modelo_avaliado', y='nota_do_juiz', text_auto='.2f', color='nota_do_juiz',
            color_continuous_scale='Viridis', title="Ranking Oficial de Notas"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_graf2:
        # Boxplot
        fig_box = px.box(
            df_filtrado, x='categoria_modelo', y='nota_do_juiz', color='categoria_modelo',
            points="all", title="Dispersão de Desempenho por Arquitetura",
            labels={'categoria_modelo': 'Arquitetura', 'nota_do_juiz': 'Nota do Juiz'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ABA 3: RISCO CLÍNICO E URGÊNCIAS (NOTA 1)
with aba3:
    st.markdown("#### 🚨 Eventos Adversos e Omissão de Urgência")
    df_criticos = df_filtrado[df_filtrado['nota_do_juiz'] == 1].copy()
    
    if df_criticos.empty:
        st.success("Nenhuma falha clínica crítica (Nota 1) encontrada para os filtros atuais!")
    else:
        st.warning(f"Atenção: Identificadas {len(df_criticos)} condutas médicas de altíssimo risco.")
        palavras_risco = ['urgência', 'emergência', 'risco', 'letal', 'óbito', 'grave', 'ignora', 'fatal', 'perigoso']
        
        def detecta_urgencia(texto):
            texto_lower = str(texto).lower()
            return any(palavra in texto_lower for palavra in palavras_risco)
            
        df_criticos['falha_urgencia'] = df_criticos['justificativa_do_juiz'].apply(detecta_urgencia)
        df_urgencia = df_criticos[df_criticos['falha_urgencia'] == True]
        
        st.error(f"Destas, {len(df_urgencia)} falharam especificamente em reconhecer a gravidade/urgência do paciente segundo o raciocínio do Juiz.")
        
        for index, row in df_urgencia.head(10).iterrows():
            with st.expander(f"⚠️ Alerta Crítico: {row['modelo_avaliado']} (Clique para Expandir)"):
                st.markdown(f"**Cenário Clínico:** {row['pergunta']}")
                st.markdown(f"**Gabarito Ouro:** {row['gabarito']}")
                st.markdown(f"**Conduta da IA:** {row['resposta_da_ia']}")
                st.markdown(f"**Auditoria do Juiz (70B):**")
                st.write(row['justificativa_do_juiz'])

# ABA 4: MATRIZ DE CORRELAÇÃO (SPEARMAN)
with aba4:
    st.markdown("#### 🧩 Matriz de Concordância (Correlação de Spearman)")
    df_pivot = df_filtrado.pivot_table(index='id_pergunta', columns='modelo_avaliado', values='nota_do_juiz')
    corr_matrix = df_pivot.corr(method='spearman')
    
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto", 
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        title="Mapa de Calor: Proximidade entre Modelos nas mesmas questões"
    )
    st.plotly_chart(fig_corr, use_container_width=True)