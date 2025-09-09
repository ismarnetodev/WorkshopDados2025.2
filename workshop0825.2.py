import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

temperatura = rd.sample(range(20, 35), 7)

temperatura_array = np.array(temperatura)

dia_da_semana = ["segunda", "terça", "quarta", "quinta", "sexta", "sabado", "domingo"]


data = {
    'Dias da semana': dia_da_semana,
    "Temperatura (°C)": temperatura
    }

df= pd.DataFrame(data)
print("\nDataFrame:\n", df)

media = np.mean(temperatura_array)
print("\ntemperatura media da semnana", media)
print("temperatura maxima:", np.max(temperatura_array))
print("temperatura minima:", np.max(temperatura_array))

plt.plot(dia_da_semana, temperatura, marker='o', label='Variação da temperatura em João Pessoa')
plt.xlabel('dias da semana')
plt.ylabel('temperatura')
plt.title('variação da temperatura')
plt.legend()
plt.grid(True)
plt.show()