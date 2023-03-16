import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

tickers = [ 'PETZ3.SA', 'VIVA3.SA','VIIA3.SA','AESB3.SA']
carteira = pd.DataFrame()
for i in tickers:
  carteira[i] = yf.download(i, period='1y')['Adj Close']

retorno = carteira.pct_change()


num_acoes = len(tickers)
num_carteiras = 10
Rf = 0

lista_retornos = []
lista_volatilidade = []
lista_pesos = []
lista_sharpe_ratio = []


for carteira in range(num_carteiras):
  #gerando pesos aleartorios para as ações
  peso = np.random.random(num_acoes)
  peso = np.round((peso / np.sum(peso)), 3)
  lista_pesos.append(peso)
  #calculo de retorno esperado
  retorno_esperado = np.sum(retorno.mean() * peso) * 264
  lista_retornos.append(retorno_esperado)
  #calculo risco
  Matrix_cov = retorno.cov() * 264
  variancia = np.dot(peso.T,    
                     np.dot(Matrix_cov, peso))
  desvio_padrao = np.sqrt(variancia)
  lista_volatilidade.append(desvio_padrao)

  #calculo Sharpe ratio
  sharpe_ratio = (retorno_esperado - Rf) / desvio_padrao
  lista_sharpe_ratio.append(sharpe_ratio)


lista_retornos = np.array(lista_retornos)
lista_volatilidade = np.array(lista_volatilidade)
lista_sharpe_ratio = np.array(lista_sharpe_ratio)


metricas = [lista_retornos, lista_volatilidade, lista_sharpe_ratio]
portifolios_df = pd.DataFrame(metricas).T
for contar,acao in enumerate(tickers):
  portifolios_df[acao+' Peso'] = [Peso[contar] for Peso in lista_pesos]
portifolios_df.columns = ['Retorno', 'Risco', 'Sharpe'] + [acao+' Peso' for acao in tickers]



min_Risco = portifolios_df.iloc[portifolios_df['Risco'].astype(float).idxmin()]
max_Retorno = portifolios_df.iloc[portifolios_df['Retorno'].astype(float).idxmax()]
max_Sharpe = portifolios_df.iloc[portifolios_df['Sharpe'].astype(float).idxmax()]

#portifolios_df.to_csv("10000portifolios_4acoes.csv")


print(portifolios_df)
print('')
print('Minimo risco')
print(min_Risco)
print('')
print('Maximo Retorno')
print(max_Retorno)
print('')
print('Max Sharpe')
print(max_Sharpe)
print('')


#grafico

plt.figure(figsize = (10,5))
plt.scatter(lista_volatilidade, lista_retornos,
            c = lista_retornos / lista_volatilidade)
plt.title('Otimização de Portifólio', fontsize = 26)
plt.xlabel('Volatilidade', fontsize = 20)
plt.ylabel('Retorno', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.colorbar(label = 'Sharpe Ratio')
plt.figure(y=[carteira.loc[min_Risco]['retorno']], 
                x=[carteira.loc[min_Risco]['volatilidade']],                  
                )
plt.show()

