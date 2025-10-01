# ⚽ Previsão do Campeonato Brasileiro

Este projeto utiliza **Machine Learning** com `scikit-learn` e `Streamlit` para prever:

* O **campeão do Brasileirão** em cada temporada.
* O **resultado de jogos individuais** com base em estatísticas.
* Além disso, oferece ferramentas de **avaliação de modelos** e **gerenciamento (salvar/carregar)**.

---

## 🆚 Diferenças entre o código antigo e o novo

### Antigo

* Apenas **Random Forest** para prever campeões e **Logistic Regression** para jogos.
* Input simples: gols, posse, chutes, escanteios, faltas.
* Avaliação limitada: só mostrava **acurácia**.
* Sem **validação cruzada** nem ajuste de hiperparâmetros.
* Menos recursos de análise (sem matriz de confusão, AUC, F1-score etc.).

### Novo

* **Novos pré-processamentos**:

  * `ColumnTransformer`, `SimpleImputer`, `StandardScaler`.
  * Criação de **features históricas** (médias de gols, % vitórias).
* **Validação mais robusta**:

  * `StratifiedKFold` + `cross_val_score`.
  * Métricas adicionais: **F1-Score, AUC-ROC, Matriz de Confusão, Classification Report**.
* **Mais abas na interface**:

  * 🔹 Previsão de Campeões
  * 🔹 Previsão de Jogos
  * 🔹 Gerenciamento de Modelos (salvar/carregar)
  * 🔹 **Análise dos Modelos** (novidade)
* **Input do usuário atualizado**: agora inclui dados históricos (média de gols, % vitórias etc.).
* Estrutura mais modular e profissional.

---

## 📂 Estrutura do Projeto

```
📦 previsao_brasileirao
 ┣ 📜 brasil.csv              # Base de dados (necessária)
 ┣ 📜 app.py                  # Código principal (Streamlit)
 ┣ 📜 modelo_campeoes.joblib  # Modelo treinado para campeões (opcional)
 ┣ 📜 modelo_jogos.joblib     # Modelo treinado para jogos (opcional)
 ┣ 📜 README.md               # Documentação
```

---

## ▶️ Como Executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/usuario/previsao_brasileirao.git
   cd previsao_brasileirao
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Coloque o arquivo `brasil.csv` na raiz do projeto.

4. Execute o Streamlit:

   ```bash
   streamlit run app.py
   ```

---

## 📊 Funcionalidades

### 🔹 Previsão de Campeões

* Treina um modelo **Random Forest** para prever o campeão de cada temporada.
* Mostra **probabilidade de cada time ser campeão**.
* Avaliação com acurácia, F1, AUC e validação cruzada.

### 🔹 Previsão de Jogos

* Treina um modelo de **Regressão Logística** com features históricas.
* Permite ao usuário inserir estatísticas de casa e visitante.
* Mostra o resultado previsto e as probabilidades.

### 🔹 Gerenciamento de Modelos

* Salvar e carregar modelos já treinados (`.joblib`).

### 🔹 Análise dos Modelos

* Exibe **matriz de confusão** e **relatório de classificação** para análise detalhada.

---

## ⚠️ Observações Importantes

* O arquivo `brasil.csv` deve conter as colunas:

  * `temporada`, `time_casa`, `time_visitante`, `resultado`
  * Além de estatísticas como: `gols_casa`, `gols_visitante`, `posse_bola_casa`, `chutes_casa`, etc.
* As features históricas são criadas automaticamente pelo código.
* Certifique-se de usar dados consistentes para melhores previsões.
