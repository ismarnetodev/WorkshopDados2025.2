# ⚽ Previsão do Campeonato Brasileiro

Este projeto é uma aplicação interativa desenvolvida em **Python** com **Streamlit** que utiliza técnicas de *Machine Learning* para prever:

1. **Campeões por temporada** do Campeonato Brasileiro
2. **Resultados de jogos individuais** (vitória da casa, empate ou vitória do visitante)

Além disso, a aplicação permite salvar e carregar os modelos treinados para reuso posterior.

---

## 🚀 Tecnologias Utilizadas

* [Python 3.9+](https://www.python.org/)
* [Streamlit](https://streamlit.io/) – Interface interativa
* [Pandas](https://pandas.pydata.org/) – Manipulação de dados
* [Scikit-learn](https://scikit-learn.org/stable/) – Modelagem de Machine Learning
* [Joblib](https://joblib.readthedocs.io/) – Persistência dos modelos

---

## 📂 Estrutura dos Dados (`brasil.csv`)

O arquivo `brasil.csv` é essencial para o funcionamento da aplicação.
Ele deve conter os registros de jogos do Campeonato Brasileiro, com a seguinte estrutura mínima:

| Coluna                 | Descrição                                                                |
| ---------------------- | ------------------------------------------------------------------------ |
| `temporada`            | Ano/temporada do campeonato                                              |
| `time_casa`            | Nome do time mandante                                                    |
| `time_visitante`       | Nome do time visitante                                                   |
| `gols_casa`            | Gols marcados pelo time da casa                                          |
| `gols_visitante`       | Gols marcados pelo time visitante                                        |
| `resultado`            | Resultado do jogo: `"vitoria_casa"`, `"empate"` ou `"vitoria_visitante"` |
| `posse_bola_casa`      | Percentual de posse de bola do time da casa                              |
| `chutes_casa`          | Número de chutes do time da casa                                         |
| `chutes_visitante`     | Número de chutes do time visitante                                       |
| `escanteios_casa`      | Escanteios a favor do time da casa                                       |
| `escanteios_visitante` | Escanteios a favor do visitante                                          |
| `faltas_casa`          | Faltas cometidas pelo time da casa                                       |
| `faltas_visitante`     | Faltas cometidas pelo time visitante                                     |

🔎 **Observação:**

* Para a aba *Previsão de Campeões*, são utilizadas as colunas de gols e resultados para calcular pontos e saldo de gols.
* Para a aba *Previsão de Jogos*, são utilizadas todas as estatísticas listadas acima.

---

## 📊 Funcionalidades

### 1. Previsão de Campeões

* Calcula os pontos e saldo de gols por temporada.
* Treina um modelo **Random Forest** para identificar o campeão.
* Mostra a **acurácia** do modelo e a probabilidade de cada time ser campeão.
* Exibe ranking e gráfico de barras por temporada.

### 2. Previsão de Resultado de Jogo

* Utiliza **Regressão Logística** para prever resultados individuais.
* Permite que o usuário insira estatísticas do jogo (gols, chutes, posse de bola, etc.).
* Retorna a previsão (vitória casa, empate ou vitória visitante) com probabilidades detalhadas.

### 3. Gerenciamento de Modelos

* Salvar modelos treinados (`.joblib`) para uso posterior.
* Carregar modelos já salvos.
* Visualizar status dos modelos disponíveis.

---

## ▶️ Como Executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/seuusuario/previsao-brasileirao.git
   cd previsao-brasileirao
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Coloque o arquivo `brasil.csv` na raiz do projeto.

5. Execute a aplicação:

   ```bash
   streamlit run app.py
   ```

---

## 📌 Requisitos

O arquivo `requirements.txt` deve conter:

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 📝 Observações

* Certifique-se de que o **`brasil.csv`** contém as colunas necessárias.
* Caso queira usar outra base de dados, adapte os nomes das colunas no código.
* Os modelos são treinados a cada execução, mas podem ser salvos e recarregados para agilizar o processo.
