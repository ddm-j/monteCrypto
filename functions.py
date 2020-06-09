import pandas as pd
import numpy as np
import yfinance as uf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_montecarlo as mc
from calendar import monthrange
import scipy as sc
from radar import radar_factory
from functools import reduce
from itertools import product
from indicators import adaptiveLaguerre

class cryptoCoins(object):

    """
    Meant to store the data of each coin that is being simulated
    including exchange price, network hashrate, block time, and
    average reward.

    """

    def __init__(self,include=None):

        self.coins = {
            "btc":{
                'unit':12,
                'block_time':800,
                'reward':13.1372
            },
            "bsv":{
                'unit':12,
                'block_time': 592,
                'reward': 12.50869
            },
            "bch":{
                'unit':12,
                'block_time': 572,
                'reward': 12.50396
            },
            "dash":{
                'unit':9,
                'block_time': 157,
                'reward': 3.11548
            },
            "btg":{
                'unit':9,
                'block_time': 588,
                'reward': 12.500973
            },
            "vtc":{
                'unit':9,
                'block_time': 143,
                'reward': 25.000229
            },
            "ltc":{
                'unit':12,
                'block_time': 156,
                'reward': 12.52304
            },
            "xmr":{
                'unit':6,
                'block_time': 118,
                'reward': 1.83437
            },
            "zec":{
                'unit':9,
                'block_time': 75,
                'reward': 6.25
            },
            "etc":{
                'unit':12,
                'block_time': 13.3,
                'reward': 4.00938
            },
            "eth":{
                'unit':12,
                'block_time': 13.33,
                'reward': 2.24811
            },
        }

        self.remove(include)

    def extract(self, d, keys):
        return dict((k, d[k]) for k in keys if k in d)

    def remove(self, include):

        self.coins = self.extract(self.coins,include)

class cryptoData(object):

    def __init__(self,coins,start,end):
        self.coins = coins

        self.getData(start, end)
        self.regression()

    def getData(self, start, end, visualize=False):

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        fields = ['date', 'HashRate', 'PriceBTC', 'PriceUSD']
        frames = []
        for coin in self.coins:

            df = pd.read_csv('Data/{0}.csv'.format(coin), usecols=fields, index_col=0, parse_dates=True)
            df.HashRate = df.HashRate*10**self.coins[coin]['unit']
            df.dropna(axis=0, inplace=True)
            frames.append(df)

        data = pd.concat(frames, keys = self.coins.keys())

        # Find the Minimum Starting Date of all of the Sets
        dates_min = []
        dates_max = []

        for coin, df in data.groupby(level=0):
            dates_min.append(df.index[0][1])
            dates_max.append(df.index[-1][1])
        idx = pd.IndexSlice

        start = start if start > max(dates_min) else max(dates_min)
        end = end if end < min(dates_max) else min(dates_max)
        self.data = data.loc[idx[:,start:end],:]

        print('Data loaded for {0} cryptocurrenices from {1} to {2}'.format(len(self.coins),start,end))

        if visualize:

            f, a = plt.subplots(len(self.coins),1)

            for i,coin in enumerate(self.coins):
                data.xs(coin).HashRate.plot(ax=a[i])

            plt.show()

    def regression(self,max_lags=90,visualize=False):

        # Determine the Regression Relationship between the Hash Rate and Exchange Rate

        models = {}
        for coin in self.coins:

            exchange = np.log(self.data.xs(coin).PriceUSD.values)
            hashrate = np.log(self.data.xs(coin).HashRate.values)

            R = []
            results = []
            for lag in np.arange(1, max_lags):

                X = sm.add_constant(exchange[:-lag])
                Y = hashrate[lag:]
                model = sm.OLS(Y,X)
                res = model.fit()
                R.append(res.rsquared)
                results.append(res)
            opt_lag = np.arange(1,max_lags)[np.argmax(R)]
            a, b = (results[np.argmax(R)]).params

            models.update({coin:[opt_lag,a,b]})

        self.regression_models = models

        print('Regression models successfully estimated')

        if visualize:
            f, a = plt.subplots(len(self.coins), 1)
            for i,coin in enumerate(self.coins):
                lag, alpha, beta = self.regression_models[coin]
                print(lag,alpha,beta)
                X = self.data.xs(coin).PriceUSD.values[:-self.regression_models[coin][0]]
                Y_fit = np.exp(alpha)*(X**beta)
                Y_org = self.data.xs(coin).HashRate.values[:-self.regression_models[coin][0]]

                # Plot Original Hashrate
                a[i].plot(Y_org,c='blue')
                a[i].set_title(coin)
                #a[i].set_ylabel('Original hashrate')

                a2 = a[i].twinx()
                # Plot Estimated Hashrate
                a2.plot(Y_fit,c='red')
                #a2.set_ylabel('Estimated Hashrate')

            plt.show()

class simulationData(object):

    def __init__(self,cryptoData,n_sims=200):

        self.n_sims = n_sims
        self.cryptoData = cryptoData
        self.coins = self.cryptoData.coins.keys()
        self.monteCarlo()

    def monteCarlo(self):

        frames = []
        hashsim = []
        for coin in self.cryptoData.coins:
            lag, a, b = self.cryptoData.regression_models[coin]
            returns = self.cryptoData.data.xs(coin).PriceUSD.pct_change()
            sims = returns.dropna().montecarlo(sims=self.n_sims)
            sims = (sims.data+1.0).cumprod()*self.cryptoData.data.xs(coin).PriceUSD.iloc[0]
            lags = sims.set_index(self.cryptoData.data.xs(coin).index[1:]).shift(-lag)
            hash = np.exp(a)*lags**b
            hashsim.append(hash)
            frames.append(sims.set_index(self.cryptoData.data.xs(coin).index[1:]))
        data = pd.concat(frames, keys = self.coins)
        hash = [i.dropna(inplace=True) for i in hashsim] # Modifies the lag dataframes inplace and returns None type
        hash = pd.concat(hashsim, keys = self.coins)

        # Find the Minimum Starting Date of all of the Sets
        dates_min = []
        dates_max = []

        for coin, df in hash.groupby(level=0):
            dates_min.append(df.index[0][1])
            dates_max.append(df.index[-1][1])

        idx = pd.IndexSlice
        start = max(dates_min)
        end = min(dates_max)
        self.simHash = hash.loc[idx[:,start:end],:]
        self.simData = data.loc[idx[:,start:end],:]

        print('Monte-Carlo Simulations Generated and Hashrate Lag Estimators Calculated')

class miningSim(object):

    def __init__(self,coins,simData):
        self.simData = simData.simData
        self.simHash = simData.simHash
        self.coins = coins.coins
        self.power_rate = 0.07

    def coin_performance(self,coin,hash_rate,power,exit="default",p=None):

        probability = hash_rate/self.simHash.xs(coin)
        blocks_mined = (3600*24/self.coins[coin]['block_time'])*probability
        reward = blocks_mined*self.coins[coin]['reward']

        if exit == "default":
            # Resample Reward to Months and Price to Months
            price = self.simData.xs(coin).copy().resample('M',convention='end').asfreq()
            reward = reward.resample('M').sum()
            revenue = (price*reward).dropna()
        elif exit == 'alag':
            # Calculate Adaptive Laguerre Filter for Each
            alags = self.simData.xs(coin).copy()
            revenue = self.simData.xs(coin).copy()
            for col in self.simData.xs(coin).columns:
                alags[col] = adaptiveLaguerre(self.simData.xs(coin)[col],length=p)
                posCross = np.where(np.diff(np.sign(alags[col])) == 2)[0] + 1
                negCross = np.where(np.diff(np.sign(alags[col])) == -2)[0] + 1
                idxCross = list(zip(posCross, negCross[1:]))
                revenue[col] = self.simData.xs(coin)[col][alags[col] < 0] * reward[col][alags[col]<0]
                revenue[col][alags[col].isnull()] = self.simData.xs(coin)[col][alags[col].isnull()] * reward[col][alags[col].isnull()]
                for idxSet in idxCross:
                    s, e = idxSet
                    rev = self.simData.xs(coin)[col][e] * reward[col].iloc[s:e].sum()
                    revenue[col].iloc[e] = rev
                if np.isnan(revenue[col].iloc[-1]):
                    idx = revenue[col].last_valid_index()+pd.DateOffset(days=1)
                    revenue[col][-1] = self.simData.xs(coin)[col][-1]*reward[col][idx:].sum()

            revenue = revenue.resample('M').sum()

        # Power Useage
        days = [monthrange(i.year,i.month)[1] for i in revenue.index]
        cost = self.power_rate*power*24*np.array(days)/1000
        cost = pd.Series(cost,index=revenue.index)

        # Profit
        profit = revenue.subtract(cost,axis='index')
        n = len(profit)
        bars, bins, _ = plt.hist(profit.sum(), bins=25)
        #plt.close()
        mode = bins[np.argmax(bars)]/n
        mean = np.mean(profit.sum())/n
        median = np.median(profit.sum())/n
        std = np.std(profit.sum()/n)

        profit_stats = {
            'mode':mode,
            'mean':mean,
            'median':median,
            'std':std,
            'average_cost':np.mean(cost),
            'profit2cost':mode/np.mean(cost),
            'rev':(mode/np.mean(cost))*np.mean(cost)+np.mean(cost)
        }
        #print(profit_stats)
        #plt.show()
        return profit_stats


    def card_performance(self,card,visualize=False):

        results = {}
        for coin in self.coins:
            res = self.coin_performance(coin,card[coin].iloc[0],card[coin].iloc[1])
            results.update({coin:res})
        results = pd.DataFrame(results)

        if visualize:

            labels = list(results.columns)
            markers = [-1,-0.5,0,0.5,1.0,1.5,2.0]
            str_markers = [str(i) for i in markers]
            data = [labels,
                    ('{0} Profit / Cost'.format(card), [
                        list(results.loc['profit2cost'])
                    ])]

            N = len(data[0])
            theta = radar_factory(N, frame='polygon')

            spoke_labels = data.pop(0)
            title, case_data = data[0]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(top=0.85, bottom=0.05)

            ax.set_rgrids([-1.0,-0.5,0,0.5,1.0,1.5,2.0])
            ax.set_title(title, position=(0.5, 1.1), ha='center')

            for d in case_data:
                line = ax.plot(theta, d)
                ax.fill(theta, d, alpha=0.25)
            ax.set_varlabels(spoke_labels)

            plt.show()

        return results


class cards(object):

    def __init__(self,exlude=None):

        load = ['radeonvii','rx580','gtx1060','gtx1070','gtx1080','gtx1080ti']
        frames = []
        for card in load:

            df = pd.read_excel('Data/Cards/{0}.xlsx'.format(card))
            df.drop(columns='col',inplace=True)
            frames.append(df)

        self.cards = pd.concat(frames,keys=load)
        self.costs = {
            'radeonvii':600,
            'rx580':140,
            'gtx1060':180,
            'gtx1070':325,
            'gtx1080':470,
            'gtx1080ti':500
        }


class rigBuilder(object):

    def __init__(self,cards,coins,simulation):
        self.card_costs = cards.costs
        self.cards = cards.cards
        self.card_names = list(set(self.cards.index.get_level_values(0).to_list()))
        self.coins = coins
        self.simulation = simulation
        self.performance = {}
        self.rig_power = 300
        self.rig_power_cost = 0.07*300*24*30/1000

    def configs(self,max_gpus=6):

        # Brute Force the Possible GPU Configurations
        configs = [np.arange(0,max_gpus+1) for name in self.card_names]
        configs = list(product(*configs))
        configs = [i for i in configs if sum(i)==max_gpus]

        return configs

    def get_performance(self):

        for name in self.card_names:
            self.performance.update({name:self.simulation.card_performance(self.cards.xs(name))})

    def rig_performance(self,config,performance):

        revs = []
        costs = []
        for i, card in enumerate(performance):

            revenue = performance[card].loc['mode']*config[i]
            costs.append(self.card_costs[card]*config[i])
            revs.append(revenue)

        df = pd.DataFrame(revs)
        df = df.sum(axis=0)-self.rig_power_cost
        df['cost'] = sum(costs)

        return df


    def test_rigs(self):

        self.get_performance()
        test_configs = self.configs(max_gpus=6)
        data = {}
        for config in test_configs:
            df = self.rig_performance(config,self.performance)
            data.update({config:df})
        data = pd.DataFrame(data).T

        for col in data.columns:
            if col != 'cost':
                data[col+'_breakeven'] = data['cost']/data[col]

        key = {}

        for i, card in enumerate(self.card_names):
            key.update({i:card})

        return data, key




coins = cryptoCoins(include=['eth'])
data = cryptoData(coins.coins, start='2019-01-01',end='2021-01-01')
monteCarlo = simulationData(data,n_sims=500)
simulation = miningSim(coins,monteCarlo)
results_alag = simulation.coin_performance('eth',529.26*10**6,1314,exit='alag',p=3)
results_monthly = simulation.coin_performance('eth',529.26*10**6,1314,exit='default',p=30)
print(results_monthly)
print(results_alag)
plt.show()

#cards = cards()
#rig = rigBuilder(cards,coins,simulation)
#data,keys = rig.test_rigs()
#data = data[(data > 0).all(1)]
#print(data.sort_values(by='etc_breakeven'))
#plt.scatter(data['cost'],data['etc_breakeven'])
#plt.show()
