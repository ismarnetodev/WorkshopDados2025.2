import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Previsão do Brasileirão", page_icon="⚽", layout="wide")

st.title("Previsão do Campeonato Brasileiro")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("brasil.csv")
        return df
    except:
        st.error("Erro ao carregar o arquivo 'brasil.csv'. Verifique se o arquivo existe.")
        return None

df = load_data()

if df is not None:
    tab1, tab2, tab3 = st.tabs(["Previsão de Campeões", "Prever Resultado de Jogo", "Gerenciar Modelos"])
    
    with tab1:
        st.header("Previsão de Campeões por Temporada")
    
        tabela = df.groupby(["temporada", "time_casa"]).agg({
            "gols_casa": "sum",
            "gols_visitante": "sum",
            "resultado": lambda x: (
                (x == "vitoria_casa").sum()*3 + (x == "empate").sum()
            )
        }).reset_index()

        tabela.rename(columns={
            "time_casa": "time",
            "gols_casa": "gols_marcados",
            "gols_visitante": "gols_sofridos",
            "resultado": "pontos"
        }, inplace=True)

        tabela["saldo_gols"] = tabela["gols_marcados"] - tabela["gols_sofridos"]

        tabela["campeao"] = tabela.groupby("temporada")["pontos"].transform(
            lambda x: (x == x.max()).astype(int)
        )

        X = tabela[["gols_marcados", "gols_sofridos", "saldo_gols", "pontos"]]
        y = tabela["campeao"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo_campeoes = RandomForestClassifier()
        modelo_campeoes.fit(X_train, y_train)

        y_pred = modelo_campeoes.predict(X_test)
        acc_campeoes = accuracy_score(y_test, y_pred)
        
        st.write(f"Acurácia do modelo de previsão de campeões: **{acc_campeoes*100:.2f}%**")

        probs = modelo_campeoes.predict_proba(X)[:,1]
        tabela["prob_campeao"] = probs

        st.subheader("Probabilidades de Campeão por Time")
        
        temporadas = sorted(tabela["temporada"].unique())
        selected_temporada = st.selectbox("Selecione a temporada:", temporadas, key="temp_campeoes")
        
        tabela_filtrada = tabela[tabela["temporada"] == selected_temporada].sort_values("prob_campeao", ascending=False)
        
        tabela_exibicao = tabela_filtrada[["time", "pontos", "gols_marcados", "gols_sofridos", "saldo_gols", "prob_campeao"]].copy()
        tabela_exibicao["prob_campeao"] = tabela_exibicao["prob_campeao"].apply(lambda x: f"{x*100:.2f}%")
        tabela_exibicao = tabela_exibicao.rename(columns={
            "time": "Time",
            "pontos": "Pontos",
            "gols_marcados": "Gols Marcados",
            "gols_sofridos": "Gols Sofridos",
            "saldo_gols": "Saldo de Gols",
            "prob_campeao": "Probabilidade de Campeão"
        })
        
        st.dataframe(tabela_exibicao, hide_index=True)
        
        st.subheader("Distribuição de Probabilidades")
        chart_data = tabela_filtrada[["time", "prob_campeao"]].set_index("time")
        st.bar_chart(chart_data)
    
    with tab2:
        st.header("Previsão de Resultado de Jogo")
        
        x = df[[
            "gols_casa", "gols_visitante", "posse_bola_casa",
            "chutes_casa", "chutes_visitante",
            "escanteios_casa", "escanteios_visitante",
            "faltas_casa", "faltas_visitante"
        ]]

        y = df["resultado"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

        modelo_jogo = LogisticRegression(max_iter=1000)
        modelo_jogo.fit(X_train, y_train)

        y_pred = modelo_jogo.predict(X_test)
        acc_jogo = accuracy_score(y_test, y_pred)

        st.write(f"Acurácia do modelo de previsão de jogos: **{acc_jogo*100:.2f}%**")
        
        st.divider()
        st.subheader("Faça sua previsão")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Time da Casa**")
            gols_casa = st.number_input("Gols do time da casa", min_value=0, step=1, key="gols_casa")
            posse_bola_casa = st.slider("Posse de bola da casa (%)", 0, 100, 50, key="posse_casa")
            chutes_casa = st.number_input("Chutes do time da casa", min_value=0, step=1, key="chutes_casa")
            escanteios_casa = st.number_input("Escanteios da casa", min_value=0, step=1, key="escanteios_casa")
            faltas_casa = st.number_input("Faltas da casa", min_value=0, step=1, key="faltas_casa")
        
        with col2:
            st.markdown("**Time Visitante**")
            gols_visitante = st.number_input("Gols do time visitante", min_value=0, step=1, key="gols_visitante")
            chutes_visitante = st.number_input("Chutes do time visitante", min_value=0, step=1, key="chutes_visitante")
            escanteios_visitante = st.number_input("Escanteios do visitante", min_value=0, step=1, key="escanteios_visitante")
            faltas_visitante = st.number_input("Faltas do visitante", min_value=0, step=1, key="faltas_visitante")

        if st.button("Prever resultado", type="primary", key="btn_prever"):
            entrada = pd.DataFrame([[
                gols_casa, gols_visitante, posse_bola_casa,
                chutes_casa, chutes_visitante,
                escanteios_casa, escanteios_visitante,
                faltas_casa, faltas_visitante
            ]], columns=x.columns)
            
            resultado_pred = modelo_jogo.predict(entrada)[0]
            resultado_final = le.inverse_transform([resultado_pred])[0]

            st.markdown("---")
            st.subheader("Resultado da Previsão")
            
            if resultado_final == "vitoria_casa":
                st.success("Vitória do time da **casa**!")
            elif resultado_final == "vitoria_visitante":
                st.success("Vitória do **visitante**!")
            else:
                st.info("O jogo terminou em **empate**.")
                
            probas = modelo_jogo.predict_proba(entrada)[0]
            st.write("Probabilidades:")
            col_prob1, col_prob2, col_prob3 = st.columns(3)
            
            with col_prob1:
                st.metric("Vitória da Casa", f"{probas[0]*100:.1f}%")
            with col_prob2:
                st.metric("Empate", f"{probas[1]*100:.1f}%")
            with col_prob3:
                st.metric("Vitória Visitante", f"{probas[2]*100:.1f}%")
    
    with tab3:
        st.header("Gerenciamento de Modelos")
        
        col_save1, col_save2 = st.columns(2)
        
        with col_save1:
            st.subheader("Salvar Modelos")
            
            if st.button("Salvar Modelo de Campeões", key="btn_save_campeoes"):
          
                modelo_campeoes_data = {
                    'model': modelo_campeoes,
                    'features': list(X.columns)
                }
                joblib.dump(modelo_campeoes_data, 'modelo_campeoes.joblib')
                st.success("Modelo de campeões salvo com sucesso!")
                
            if st.button("Salvar Modelo de Jogos", key="btn_save_jogos"):
          
                modelo_jogo_data = {
                    'model': modelo_jogo,
                    'label_encoder': le,
                    'features': list(x.columns)
                }
                joblib.dump(modelo_jogo_data, 'modelo_jogos.joblib')
                st.success("Modelo de jogos salvo com sucesso!")
                
            if st.button("Salvar Todos os Modelos", key="btn_save_all"):
           
                modelo_campeoes_data = {
                    'model': modelo_campeoes,
                    'features': list(X.columns)
                }
                joblib.dump(modelo_campeoes_data, 'modelo_campeoes.joblib')
                
                modelo_jogo_data = {
                    'model': modelo_jogo,
                    'label_encoder': le,
                    'features': list(x.columns)
                }
                joblib.dump(modelo_jogo_data, 'modelo_jogos.joblib')
                
                st.success("Todos os modelos salvos com sucesso!")
        
        with col_save2:
            st.subheader("Carregar Modelos")
            
            if st.button("📂 Carregar Modelo de Campeões", key="btn_load_campeoes"):
                if os.path.exists('modelo_campeoes.joblib'):
                    modelo_carregado = joblib.load('modelo_campeoes.joblib')
                    modelo_campeoes = modelo_carregado['model']
                    st.success("Modelo de campeões carregado com sucesso!")
                else:
                    st.error("Arquivo do modelo de campeões não encontrado!")
                    
            if st.button("📂 Carregar Modelo de Jogos", key="btn_load_jogos"):
                if os.path.exists('modelo_jogos.joblib'):
                    modelo_carregado = joblib.load('modelo_jogos.joblib')
                    modelo_jogo = modelo_carregado['model']
                    le = modelo_carregado['label_encoder']
                    st.success("Modelo de jogos carregado com sucesso!")
                else:
                    st.error(" Arquivo do modelo de jogos não encontrado!")
        

        st.subheader("Modelos Salvos")
        if os.path.exists('modelo_campeoes.joblib'):
            st.info("📁 modelo_campeoes.joblib - Disponível")
        else:
            st.warning("📁 modelo_campeoes.joblib - Não encontrado")
            
        if os.path.exists('modelo_jogos.joblib'):
            st.info("📁 modelo_jogos.joblib - Disponível")
        else:
            st.warning("📁 modelo_jogos.joblib - Não encontrado")

else:
    st.info("Para usar esta aplicação, faça upload de um arquivo CSV chamado 'brasil.csv' com os dados do Brasileirão.")

st.sidebar.title("Sobre")
st.sidebar.info("""
Esta aplicação utiliza machine learning para:
- Prever os campeões do Brasileirão
- Prever resultados de jogos individuais

Os modelos usados são:
- Random Forest para previsão de campeões
- Regressão Logística para previsão de jogos

Use a aba 'Gerenciar Modelos' para salvar e carregar os modelos treinados.
""")

st.sidebar.warning("Certifique-se de que o arquivo 'brasil.csv' possui as colunas necessárias para que os modelos funcionem corretamente.")