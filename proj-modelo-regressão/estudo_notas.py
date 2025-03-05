# %%

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("../data/dados_estudo_nota.xlsx")
df
# %%
# Criando gráfico dos estudos

plt.plot(df["estudo"], df["nota"], 'o')
plt.grid(True)
plt.title("Relação Nota vs Estudo")
plt.ylim(0, 11)
plt.xlim(0, 11)
plt.xlabel("Estudo")
plt.ylabel("Nota")
plt.show()

# %%

# Modelo de Regressão
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df[["estudo"]], df["nota"])

# %%

a, b = reg.intercept_, reg.coef_[0]
print(f"a={a}; b={b}")

# %%
X = df[["estudo"]].drop_duplicates()
y_estimado = reg.predict(X)
y_estimado

plt.plot(df["estudo"], df["nota"], 'o')
plt.plot(X, y_estimado, '-')
plt.grid(True)
plt.title("Relação Nota vs Estudo")
plt.ylim(0, 11)
plt.xlim(0, 11)
plt.xlabel("Estudo")
plt.ylabel("Nota")
plt.show()

# %%
# Arvore de decisãp
from sklearn import tree
arvore = tree.DecisionTreeRegressor(max_depth=2)
arvore.fit(df[["estudo"]], df["nota"])

y_estimado_arvore = arvore.predict(X)

plt.figure(dpi=500)
plt.plot(df["estudo"], df["nota"], 'o')
plt.plot(X, y_estimado, '-')
plt.plot(X, y_estimado_arvore, '-')
plt.grid(True)
plt.title("Relação Nota vs Estudo")
plt.ylim(0, 11)
plt.xlim(0, 11)
plt.xlabel("Estudo")
plt.ylabel("Nota")
plt.legend(["Observações", "Regressão Linear", "Árvore de Decisão"])
plt.show()
# %%
