import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title="Previsão do Brasileirão",
                   page_icon="⚽", layout="wide")

st.title("Previsão do Campeonato Brasileiro")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("brasil.csv")
        
        colunas_necessarias = ['temporada', 'time_casa', 'time_visitante', 'resultado']
        for coluna in colunas_necessarias:
            if coluna not in df.columns:
                st.error(f"Coluna '{coluna}' não encontrada no arquivo CSV.")
                return None
                
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo 'brasil.csv': {e}")
        return None

def criar_features_historicas(df):
    df = df.copy()
    
    for time in df['time_casa'].unique():
        jogos_casa = df[df['time_casa'] == time]
        if len(jogos_casa) > 0:
            df.loc[df['time_casa'] == time, 'media_gols_casa'] = jogos_casa['gols_casa'].mean()
            df.loc[df['time_casa'] == time, 'vitorias_casa'] = (jogos_casa['resultado'] == 'vitoria_casa').mean()
        
        jogos_visitante = df[df['time_visitante'] == time]
        if len(jogos_visitante) > 0:
            df.loc[df['time_visitante'] == time, 'media_gols_visitante'] = jogos_visitante['gols_visitante'].mean()
            df.loc[df['time_visitante'] == time, 'vitorias_visitante'] = (jogos_visitante['resultado'] == 'vitoria_visitante').mean()
    
    df.fillna({
        'media_gols_casa': 0,
        'media_gols_visitante': 0,
        'vitorias_casa': 0,
        'vitorias_visitante': 0
    }, inplace=True)
    
    return df

def avaliar_modelo(modelo, X_test, y_test, y_pred, y_proba=None):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    st.write(f"**Acurácia:** {accuracy*100:.2f}%")
    st.write(f"**F1-Score:** {f1*100:.2f}%")
    
    if y_proba is not None and len(np.unique(y_test)) > 1:
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            st.write(f"**AUC-ROC:** {auc*100:.2f}%")
        except:
            pass
    
    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=[f'Real {i}' for i in np.unique(y_test)],
                        columns=[f'Pred {i}' for i in np.unique(y_test)])
    st.dataframe(cm_df)
    
    st.subheader("Relatório de Classificação")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

df = load_data()

if df is not None:
    st.info("Criando features históricas...")
    df_com_features = criar_features_historicas(df)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Previsão de Campeões",
        "Prever Resultado de Jogo", 
        "Gerenciar Modelos",
        "Análise dos Modelos"
    ])

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

        st.write(f"**Distribuição das classes:** {y.value_counts().to_dict()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        pipeline_campeoes = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ))
        ])

        cv_scores = cross_val_score(pipeline_campeoes, X_train, y_train, 
                                  cv=StratifiedKFold(n_splits=5), 
                                  scoring='accuracy')
        
        st.write(f"**Acurácia na Validação Cruzada:** {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

        pipeline_campeoes.fit(X_train, y_train)

        y_pred = pipeline_campeoes.predict(X_test)
        y_proba = pipeline_campeoes.predict_proba(X_test)

        st.subheader("Desempenho do Modelo de Campeões")
        avaliar_modelo(pipeline_campeoes, X_test, y_test, y_pred, y_proba)

        probs = pipeline_campeoes.predict_proba(X)[:, 1]
        tabela["prob_campeao"] = probs

        st.subheader("Probabilidades de Campeão por Time")

        temporadas = sorted(tabela["temporada"].unique())
        selected_temporada = st.selectbox(
            "Selecione a temporada:", temporadas, key="temp_campeoes")

        tabela_filtrada = tabela[tabela["temporada"] == selected_temporada].sort_values(
            "prob_campeao", ascending=False)

        tabela_exibicao = tabela_filtrada[[
            "time", "pontos", "gols_marcados", "gols_sofridos", "saldo_gols", "prob_campeao"]].copy()
        tabela_exibicao["prob_campeao"] = tabela_exibicao["prob_campeao"].apply(
            lambda x: f"{x*100:.2f}%")
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

        features_corretas = [
            "media_gols_casa", "media_gols_visitante",
            "vitorias_casa", "vitorias_visitante",
            "posse_bola_casa", "chutes_casa", "chutes_visitante",
            "escanteios_casa", "escanteios_visitante",
            "faltas_casa", "faltas_visitante"
        ]

        features_disponiveis = [f for f in features_corretas if f in df_com_features.columns]
        
        if len(features_disponiveis) < 4:
            st.error("Features insuficientes para treinar o modelo. Verifique os dados.")
        else:
            x = df_com_features[features_disponiveis]
            y = df_com_features["resultado"]

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

            numeric_features = features_disponiveis
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ])

            pipeline_jogo = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])

            cv_scores_jogo = cross_val_score(pipeline_jogo, X_train, y_train, 
                                           cv=StratifiedKFold(n_splits=5), 
                                           scoring='accuracy')
            
            st.write(f"**Acurácia na Validação Cruzada:** {cv_scores_jogo.mean()*100:.2f}% (±{cv_scores_jogo.std()*100:.2f}%)")

            pipeline_jogo.fit(X_train, y_train)

            y_pred_jogo = pipeline_jogo.predict(X_test)
            y_proba_jogo = pipeline_jogo.predict_proba(X_test)

            st.subheader("Desempenho do Modelo de Jogos")
            avaliar_modelo(pipeline_jogo, X_test, y_test, y_pred_jogo, y_proba_jogo)

            st.divider()
            st.subheader("Faça sua previsão")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Time da Casa**")
                posse_bola_casa = st.slider("Posse de bola da casa (%)", 0, 100, 50, key="posse_casa")
                chutes_casa = st.number_input("Chutes do time da casa", min_value=0, value=10, step=1, key="chutes_casa")
                escanteios_casa = st.number_input("Escanteios da casa", min_value=0, value=5, step=1, key="escanteios_casa")
                faltas_casa = st.number_input("Faltas da casa", min_value=0, value=15, step=1, key="faltas_casa")
                media_gols_casa = st.number_input("Média de gols em casa", min_value=0.0, value=1.5, step=0.1, key="media_casa")

            with col2:
                st.markdown("**Time Visitante**")
                chutes_visitante = st.number_input("Chutes do time visitante", min_value=0, value=8, step=1, key="chutes_visitante")
                escanteios_visitante = st.number_input("Escanteios do visitante", min_value=0, value=4, step=1, key="escanteios_visitante")
                faltas_visitante = st.number_input("Faltas do visitante", min_value=0, value=16, step=1, key="faltas_visitante")
                media_gols_visitante = st.number_input("Média de gols fora", min_value=0.0, value=1.2, step=0.1, key="media_visitante")
                vitorias_visitante = st.slider("% Vitórias fora", 0, 100, 30, key="vitorias_visitante")

            if st.button("Prever resultado", type="primary", key="btn_prever"):
                entrada_data = {}
                for feature in features_disponiveis:
                    if 'casa' in feature:
                        if 'media_gols' in feature:
                            entrada_data[feature] = media_gols_casa
                        elif 'vitorias' in feature:
                            entrada_data[feature] = vitorias_visitante / 100  
                        elif 'posse' in feature:
                            entrada_data[feature] = posse_bola_casa
                        elif 'chutes' in feature:
                            entrada_data[feature] = chutes_casa
                        elif 'escanteios' in feature:
                            entrada_data[feature] = escanteios_casa
                        elif 'faltas' in feature:
                            entrada_data[feature] = faltas_casa
                    elif 'visitante' in feature:
                        if 'media_gols' in feature:
                            entrada_data[feature] = media_gols_visitante
                        elif 'vitorias' in feature:
                            entrada_data[feature] = vitorias_visitante / 100
                        elif 'chutes' in feature:
                            entrada_data[feature] = chutes_visitante
                        elif 'escanteios' in feature:
                            entrada_data[feature] = escanteios_visitante
                        elif 'faltas' in feature:
                            entrada_data[feature] = faltas_visitante

                entrada = pd.DataFrame([list(entrada_data.values())], columns=features_disponiveis)

                try:
                    resultado_pred = pipeline_jogo.predict(entrada)[0]
                    resultado_final = le.inverse_transform([resultado_pred])[0]
                    probas = pipeline_jogo.predict_proba(entrada)[0]

                    st.markdown("---")
                    st.subheader("Resultado da Previsão")

                    if resultado_final == "vitoria_casa":
                        st.success("Vitória do time da **casa**!")
                    elif resultado_final == "vitoria_visitante":
                        st.success("Vitória do **visitante**!")
                    else:
                        st.info("O jogo terminou em **empate**.")

                    st.write("Probabilidades:")
                    col_prob1, col_prob2, col_prob3 = st.columns(3)

                    with col_prob1:
                        st.metric("Vitória da Casa", f"{probas[0]*100:.1f}%")
                    with col_prob2:
                        st.metric("Empate", f"{probas[1]*100:.1f}%")
                    with col_prob3:
                        st.metric("Vitória Visitante", f"{probas[2]*100:.1f}%")

                except Exception as e:
                    st.error(f"Erro na previsão: {e}")

    with tab3:
        st.header("Gerenciamento de Modelos")

        col_save1, col_save2 = st.columns(2)

        with col_save1:
            st.subheader("Salvar Modelos")

            if st.button("💾 Salvar Modelo de Campeões"):
                try:
                    modelo_campeoes_data = {
                        'model': pipeline_campeoes,
                        'features': list(X.columns),
                        'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'metadados': {
                            'tipo': 'RandomForest',
                            'accuracy_cv': float(cv_scores.mean()),
                            'n_amostras': len(X_train)
                        }
                    }
                    joblib.dump(modelo_campeoes_data, 'modelo_campeoes.joblib')
                    st.success("Modelo de campeões salvo com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao salvar: {e}")

            if st.button("💾 Salvar Modelo de Jogos", key="btn_save_jogos"):
                try:
                    modelo_jogo_data = {
                        'model': pipeline_jogo,
                        'label_encoder': le,
                        'features': features_disponiveis,
                        'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'metadados': {
                            'tipo': 'LogisticRegression',
                            'accuracy_cv': float(cv_scores_jogo.mean()),
                            'n_amostras': len(X_train)
                        }
                    }
                    joblib.dump(modelo_jogo_data, 'modelo_jogos.joblib')
                    st.success("Modelo de jogos salvo com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao salvar: {e}")

            if st.button("💾 Salvar Todos os Modelos", key="btn_save_all"):
                try:
     
                    modelo_campeoes_data = {
                        'model': pipeline_campeoes,
                        'features': list(X.columns),
                        'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    joblib.dump(modelo_campeoes_data, 'modelo_campeoes.joblib')
                    
                    modelo_jogo_data = {
                        'model': pipeline_jogo,
                        'label_encoder': le,
                        'features': features_disponiveis,
                        'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    joblib.dump(modelo_jogo_data, 'modelo_jogos.joblib')
                    
                    st.success("Todos os modelos salvos com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao salvar: {e}")

        with col_save2:
            st.subheader("Carregar Modelos")
            
            if st.button("📂 Carregar Modelo de Campeões", key="btn_load_campeoes"):
                if os.path.exists('modelo_campeoes.joblib'):
                    try:
                        modelo_carregado = joblib.load('modelo_campeoes.joblib')
                        pipeline_campeoes = modelo_carregado['model']
                        st.success("Modelo de campeões carregado com sucesso!")
                        if 'metadados' in modelo_carregado:
                            st.info(f"Treinado em: {modelo_carregado.get('metadados', {}).get('data_treinamento', 'N/A')}")
                    except Exception as e:
                        st.error(f"Erro ao carregar: {e}")
                else:
                    st.error("Arquivo do modelo de campeões não encontrado!")
                    
            if st.button("📂 Carregar Modelo de Jogos", key="btn_load_jogos"):
                if os.path.exists('modelo_jogos.joblib'):
                    try:
                        modelo_carregado = joblib.load('modelo_jogos.joblib')
                        pipeline_jogo = modelo_carregado['model']
                        le = modelo_carregado['label_encoder']
                        st.success("Modelo de jogos carregado com sucesso!")
                        if 'metadados' in modelo_carregado:
                            st.info(f"Treinado em: {modelo_carregado.get('metadados', {}).get('data_treinamento', 'N/A')}")
                    except Exception as e:
                        st.error(f"Erro ao carregar: {e}")
                else:
                    st.error("Arquivo do modelo de jogos não encontrado!")

        st.subheader("Modelos Salvos")
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            if os.path.exists('modelo_campeoes.joblib'):
                try:
                    modelo_info = joblib.load('modelo_campeoes.joblib')
                    st.success("📁 modelo_campeoes.joblib")
                    if 'metadados' in modelo_info:
                        st.write(f"Accuracy CV: {modelo_info['metadados'].get('accuracy_cv', 'N/A'):.3f}")
                except:
                    st.warning("📁 modelo_campeoes.joblib (corrompido)")
            else:
                st.warning("📁 modelo_campeoes.joblib - Não encontrado")
                
        with col_info2:
            if os.path.exists('modelo_jogos.joblib'):
                try:
                    modelo_info = joblib.load('modelo_jogos.joblib')
                    st.success("📁 modelo_jogos.joblib")
                    if 'metadados' in modelo_info:
                        st.write(f"Accuracy CV: {modelo_info['metadados'].get('accuracy_cv', 'N/A'):.3f}")
                except:
                    st.warning("📁 modelo_jogos.joblib (corrompido)")
            else:
                st.warning("📁 modelo_jogos.joblib - Não encontrado")

    with tab4:
        st.header("Análise Detalhada dos Modelos")
        
        col_ana1, col_ana2 = st.columns(2)
        
        with col_ana1:
            st.subheader("Modelo de Campeões")
            if 'pipeline_campeoes' in locals():
                if hasattr(pipeline_campeoes.named_steps['classifier'], 'feature_importances_'):
                    importances = pipeline_campeoes.named_steps['classifier'].feature_importances_
                    features = X.columns
                    feature_imp_df = pd.DataFrame({
                        'Feature': features,
                        'Importância': importances
                    }).sort_values('Importância', ascending=False)
                    
                    st.write("**Importância das Features:**")
                    st.dataframe(feature_imp_df, hide_index=True)
                    
                    st.bar_chart(feature_imp_df.set_index('Feature'))
        
        with col_ana2:
            st.subheader("Modelo de Jogos")
            if 'pipeline_jogo' in locals():
                if hasattr(pipeline_jogo.named_steps['classifier'], 'coef_'):
                    coefs = pipeline_jogo.named_steps['classifier'].coef_
                    features_jogo = features_disponiveis
                    
                    st.write("**Coeficientes do Modelo (média por classe):**")
                    coef_medio = np.mean(np.abs(coefs), axis=0)
                    coef_df = pd.DataFrame({
                        'Feature': features_jogo,
                        'Coeficiente Médio': coef_medio
                    }).sort_values('Coeficiente Médio', ascending=False)
                    
                    st.dataframe(coef_df, hide_index=True)

else:
    st.info("Para usar esta aplicação, faça upload de um arquivo CSV chamado 'brasil.csv' com os dados do Brasileirão.")

st.sidebar.title("Sobre")
st.sidebar.info("""
Esta aplicação utiliza machine learning para:
- Prever os campeões do Brasileirão
- Prever resultados de jogos individuais

**Melhorias implementadas:**
Correção de vazamento de dados
Validação cruzada
Pipeline robusto com pré-processamento
Múltiplas métricas de avaliação
Features históricas
Análise detalhada dos modelos

**Modelos:**
- Random Forest para previsão de campeões
- Regressão Logística para previsão de jogos
""")

st.sidebar.warning("""
Certifique-se de que o arquivo 'brasil.csv' possui as colunas necessárias:
- temporada, time_casa, time_visitante, resultado
- gols_casa, gols_visitante
- estatísticas do jogo
""")