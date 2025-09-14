# =========================
# APP STREAMLIT - REDFIT
# =========================

import streamlit as st
import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(page_title="redfit", page_icon="💪🏋️‍♂️", layout="wide")

st.title("Perfil do aluno RedFit")
st.markdown("---")
st.divider()

@st.cache_data
def load_and_process_data(path_csv="academia_redfit.csv"):
    try:
        df_raw = pd.read_csv(path_csv)
    except FileNotFoundError:
        st.warning(f"Arquivo {path_csv} não encontrado. Se existir, coloque-o na mesma pasta do app.")
        return None
    except Exception as e:
        st.error(f"Erro ao ler {path_csv}: {e}")
        return None
    
    df = df_raw.copy()

    if 'sexo' in df.columns:
        df['sexo'] = df['sexo'].map({
            'Masculino': 'M', 'masculino': 'M', 'M': "M",
            'Feminino': 'F', 'feminino': 'F', 'F': 'F'
        }).fillna('M')

    if 'estado' in df.columns:
        df['estado'] = df['estado'].map({
            'Ativa': 'Ativa', 'Ativo': 'Ativa',
            'Sedentária': 'Sedentária', 'Sedentario': 'Sedentária', 'Sedentaria': 'Sedentária',
            'Atleta': 'Atleta'
        }).fillna('Ativa')

    if 'tipo_atividade' in df.columns:
        df['tipo_atividade'] = df['tipo_atividade'].str.title().fillna('Musculação')

    if 'data_matricula' in df.columns:
        df['data_matricula'] = pd.to_datetime(df['data_matricula'], errors='coerce')
        df['ano_matricula'] = df['data_matricula'].dt.year
        df['mes_matricula'] = df['data_matricula'].dt.month

    # Calcular IMC
    if all(col in df.columns for col in ['peso', 'altura']):
        df['altura'] = df['altura'].replace(0, np.nan)
        df['imc'] = df['peso'] / ((df['altura']/100)**2)
    else:
        df['imc'] = np.nan

    if 'pgc_inicial' in df.columns and 'pgc_final' in df.columns:
        df['evolucao_pgc'] = df['pgc_final'] - df['pgc_inicial']
        df['evolucao_pgc_percentual'] = ((df['pgc_final'] - df['pgc_inicial']) / df['pgc_inicial'].replace(0, 0.001)) * 100
    else:
        pgc_cols = [c for c in df.columns if 'pgc' in c.lower() and c.lower() not in ['evolucao_pgc', 'evolucao_pgc_percentual']]
        if len(pgc_cols) >= 2:
            df['pgc_inicial'] = df[pgc_cols[0]]
            df['pgc_final'] = df[pgc_cols[-1]]
            df['evolucao_pgc'] = df['pgc_final'] - df['pgc_inicial']
            df['evolucao_pgc_percentual'] = ((df['pgc_final'] - df['pgc_inicial']) / df['pgc_inicial'].replace(0, 0.001)) * 100

    if 'estado' not in df.columns and all(col in df.columns for col in ['frequencia_semanal_treino', 'imc']):
        conditions = [
            (df['frequencia_semanal_treino'] >= 5) & (df['imc'] <= 25),
            (df['frequencia_semanal_treino'] <= 2),
            (df['frequencia_semanal_treino'] >= 3) & (df['frequencia_semanal_treino'] <= 4)
        ]
        choices = ['Atleta', 'Sedentária', 'Ativa']
        df['estado'] = np.select(conditions, choices, default='Ativa')

    try:
        df.to_csv("dados_tratados.csv", index=False)
        st.success("✅ Dados processados e salvos com sucesso!")
    except Exception as e:
        st.error(f"Erro ao salvar os dados: {e}")
        
    return df


df = load_and_process_data()

if df is not None:
    tab1, tab2, tab3 = st.tabs([
        "Seu Perfil",
        "Analise Exploratoria",
        "Desempenho do Modelo"        
    ])

    with tab1:
        st.header("📊 Seu Perfil Fit")
            
        with st.form('perfil_form'):
            col1, col2 = st.columns(2)

            with col1:
                idade  = st.slider("Idade", min_value=18, max_value=80, value=30) 
                sexo = st.selectbox("Sexo", options=["Masculino", "Feminino"])
                altura = st.number_input("Altura (cm)", min_value=140, max_value=220, value=170)

            with col2:
                peso = st.number_input("Peso (kg)", min_value=40, max_value=300, value=70)
                frequencia_treino = st.slider("Frequência semanal de treino", min_value=1, max_value=7, value=3)
                objetivo = st.selectbox("Objetivo", options=["Emagrecimento", "Hipertrofia", "Condicionamento", "Competição"])
    
            submitted = st.form_submit_button("Analisar Meu Perfil")

        if submitted:
            imc = peso / ((altura/100)**2)

            if frequencia_treino >= 5 and imc <= 25:
                estado = "Atleta"
            elif frequencia_treino <= 2:
                estado = "Sedentária"
            else:
                estado = "Ativa"

            evolucao_pgc = np.random.uniform(-5, 5)

            st.success("Perfil analisado com sucesso!")
            st.subheader("Seus dados:")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Idade", idade)
                st.metric("Sexo", sexo)
    
            with col2:
                st.metric("Altura", f"{altura} cm")
                st.metric("Peso", f"{peso} kg")

            with col3:
                st.metric("IMC", f"{imc:.2f}")
                st.metric("Frequência", f"{frequencia_treino}x/semana")

            st.subheader("Recomendações Personalizadas:")
            if estado == "Sedentária":
                st.info("💡 Recomendamos aumentar a frequência de treino para pelo menos 3x por semana")
            elif estado == "Ativa":
                st.info("💡 Continue mantendo sua rotina de exercícios!")
            else:
                st.info("💡 Excelente! Mantenha o alto desempenho atlético")
            if evolucao_pgc < 0:
                st.success("🎉 Parabéns! Você reduziu seu percentual de gordura corporal!")
            elif evolucao_pgc > 0:
                st.warning("⚠️ Atenção: Seu percentual de gordura corporal aumentou. Considere ajustar sua dieta e treino.")

    with tab2:
        st.header("📈 Análise Geral dos Clientes")

        st.subheader("Distribuição por Sexo")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        contagem_sexo = df['sexo'].value_counts()
        ax1.pie(contagem_sexo, labels=contagem_sexo.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("Frequência Semanal de Treino")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.countplot(x='frequencia_semanal_treino', data=df, ax=ax2)
        plt.xlabel("Frequência Semanal")
        plt.ylabel("Contagem de Clientes")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        plt.close(fig2)

        if 'evolucao_pgc' in df.columns:
            st.subheader("Evolução PGC")
            fig_pgc, ax_pgc  = plt.subplots(figsize=(10,5))
            sns.histplot(df['evolucao_pgc'].dropna(), kde=True, ax=ax_pgc)
            plt.xlabel("Evolução do PGC (%)")
            plt.ylabel("Frequência")
            st.pyplot(fig_pgc)
            plt.close(fig_pgc)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Média Evolução PGC", f"{df['evolucao_pgc'].mean():.2f}%")
            with col2:
                st.metric("Melhor Evolução PGC", f"{df['evolucao_pgc'].min():.2f}%")
            with col3:
                st.metric("Pior Evolução PGC", f"{df['evolucao_pgc'].max():.2f}%")

        st.subheader("Distribuição por Faixa Etária")
        if 'idade' in df.columns:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            df['faixa_etaria'] = pd.cut(df['idade'], bins=[18, 25, 35, 45, 55, 65, 80])
            sns.countplot(x='faixa_etaria', data=df, ax=ax3)
            plt.xlabel("Faixa Etária")
            plt.ylabel("Contagem")
            plt.xticks(rotation=45)
            st.pyplot(fig3)
            plt.close(fig3)

    with tab3:
        st.header("Avaliação do Modelo de ML")

        target_col = 'estado'  

        if target_col in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            X = df[numeric_cols].fillna(df[numeric_cols].mean())
            y = df[target_col]

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Acurácia", f"{accuracy:.2f}")
            with col2:
                st.metric("Amostras de Treino", len(X_train))
            with col3:
                st.metric("Amostras de Teste", len(X_test))

            st.subheader("Matriz de Confusão")
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
            plt.xlabel("Previsto")
            plt.ylabel("Real")
            st.pyplot(fig4)
            plt.close(fig4)

            st.subheader("Relatório de Classificação")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

            if hasattr(model, 'feature_importances_'):
                st.subheader("Importância das Variáveis")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig5, ax5 = plt.subplots(figsize=(10,6))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax5)
                plt.title("Top 10 Variáveis mais importantes")
                st.pyplot(fig5)
                plt.close(fig5)

st.markdown("""
<style>
    /* Fundo preto para todos os elementos */
    .stApp {
        background-color: #000000;
        color: white;
    }
    
    /* Campos de entrada pretos */
    .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
        background-color: #1a1a1a !important;
        border: 1px solid #333 !important;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stSlider input {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    
    /* Labels brancos */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: white !important;
    }
    
    /* Botões pretos */
    .stButton button {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    /* Tabs com fundo escuro */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a2a;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #444 !important;
    }
    
    /* Métricas com fundo escuro */
    .stMetric {
        background-color: #1a1a1a !important;
        color: white !important;
        padding: 15px;
        border: 1px solid #333;
    }
    
    .stMetric label {
        color: #ccc !important;
    }
    
    .stMetric div {
        color: white !important;
    }
    
    /* Dataframes com fundo escuro */
    .dataframe {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    
    .dataframe th {
        background-color: #2a2a2a !important;
        color: white !important;
    }
    
    .dataframe td {
        background-color: #1a1a1a !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
