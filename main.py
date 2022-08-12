import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas_datareader as web
import seaborn as sns


coin = str('BTC') # choose coin here ex: BTC, ETH, SOl, BNB.......
f = requests.get(f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit=2000").json()['Data']['Data']
g = pd.DataFrame(f)
df = g[['time','high', 'low', 'volumeto', 'close']]
df['time'] = pd.to_datetime(df['time'], unit='s')


dataset =  df.copy()


# PLOT EMPERICAL QUANTILES
#dataset['daily_change'] = dataset['close'].pct_change()


round(dataset['daily_change'],2).quantile(0.05)
sns.despine(left=True)
sns.displot(dataset['daily_change'], color="green")
plt.title('daily change')
plt.grid(True)
plt.show()

days  = int(252) #lookback start date

#  PLOT  VIX OF TICKER
dataset['vix'] = dataset['daily_change'].rolling(days).std()*(days**0.5)
dataset['vix'].plot(figsize=(10,5), grid=True)
plt.title('VIX')
plt.show()



dataset  = dataset.sort_values(by='time', ascending=True)
print(dataset.loc[[len(dataset)-days]]) #get close price printed and replace {startPrice} below with it



startPrice =  49243.39 # copy  close price

dt = 1/days
mu = dataset['daily_change'].mean()
sigma = dataset['daily_change'].std()

# perform Monte-Carlo 
def carlo(startPrice, days, mu,  sigma):
    price = np.zeros(days)
    price[0] = startPrice
    shock  =  np.zeros(days)
    drift = np.zeros(days)
    for  x in range(1,days):
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] *  (drift[x] + shock[x]))

    return price

plt.plot(carlo(startPrice,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('price USD')
plt.title('Monte-Carlo Sim')

plt.show()

# RUN MONTE CARLO SIM 10K times for 

runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = carlo(startPrice,days,mu,sigma)[days-1]
q = np.percentile(simulations,1)
plt.hist(simulations, bins = 200)
plt.figtext(0.6,0.8,s="Start price: $%.2f" %startPrice)
plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())
plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (startPrice -q,))
plt.figtext(0.15,0.6, "q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title(f"Final price distribution for {coin} after {days} days"  , weight='bold')
plt.show()

