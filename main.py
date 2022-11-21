import numpy as np
import pandas as pd 

from scipy.special import log_softmax, softmax, psi
from scipy.stats import beta, gamma
from scipy.optimize import minimize


# --------- models ---------- #

class rl:
    priors = [gamma(a=1, scale=5), beta(a=1.2, b=1.2)]
    bnds   = [(0, 50), (0, 1)]
    pbnds  = [(0, 10), (0,.5)]
    n_param = len(bnds)

    def __init__(self, nA):
        self.nA   = nA
        
    def sim(self, params, T=100, R=[.2, .8]):
 
        self.v = np.zeros([self.nA,])
        data = {'act': [], 'rew': [], 'trial': []}

        # decompose parameters
        self.beta = params[0]
        self.alpha = params[1]

        for t in range(T):

            p = softmax(self.beta*self.v)
            c = np.random.choice(self.nA, p=p)
            r = 1*(np.random.rand() < R[c])
            self.v[c] += self.alpha*(r-self.v[c])
            data['trial'].append(t)
            data['act'].append(c)
            data['rew'].append(r)
        
        return pd.DataFrame.from_dict(data)

    def fit(self, data, seed, init=None, verbose=False):
        '''Fit the parameter using optimization 
        '''
        # get bounds and possible bounds 
        bnds  = self.bnds
        pbnds = self.pbnds

        # Init params
        if init:
            # if there are assigned params
            param0 = init
        else:
            # random init from the possible bounds 
            rng = np.random.RandomState(seed)
            param0 = [pbnd[0] + (pbnd[1] - pbnd[0]
                     ) * rng.rand() for pbnd in pbnds]
                     
        ## Fit the params 
        if verbose: print('init with params: ', param0) 
        res = minimize(self.loss_fn, param0, args=(data), method='L-BFGS-B',
                        bounds=bnds, options={'disp': verbose})
        if verbose: print(f'''  Fitted params: {res.x}, 
                    MLE loss: {res.fun}''')

        return res

    def loss_fn(self, params, data):
        tot_loss = self._loglike(params, data) \
                + self._logprior(params)
        return tot_loss
                    
    def _loglike(self, params, data):

        self.v = np.zeros([self.nA,])
        nll = 0 

        # decompose parameters
        self.beta = params[0]
        self.alpha = params[1]

        for _, row in data.iterrows():

            act = row['act']
            rew = row['rew']

            log_p = log_softmax(self.beta*self.v)
            nll -= log_p[act]
            self.v[act] += self.alpha*(rew-self.v[act])

        return nll 

    def _logprior(self, params):
        logprior = 0
        for prior, param in zip(self.priors, params):
            logprior += -np.max([prior.logpdf(param), -1e12])
        return logprior


#--------- get data  -------- #
def get_data():
    params = [[8, 0.1], [6, 0.2], [2, 0.1], [5, 0.3]]
    sim_data = {}

    for s in range(4):
        model = rl(2)
        sim_data[s] = model.sim(params[s])

    return sim_data 

if __name__ == '__main__':

    sim_data = get_data()
    model = rl(2)
    seed = 2021
    fields = ['logpost', 'param', 'H', 'aic', 'bic', 'n_param']
    res_fit = {k: [] for k in fields}

    for s in range(4):
        results = [model.fit(sim_data[s], seed+i) for i in range(10)]
        idx = np.argmin([res.fun for res in results])
        res_opt = results[idx]
        res_fit['logpost'].append(-res_opt.fun)
        res_fit['param'].append(res_opt.x)
        res_fit['n_param'] = rl.n_param
        res_fit['H'].append(np.linalg.inv(res_opt.hess_inv.todense()))
        res_fit['aic'].append(2*rl.n_params + 2*res_opt.fun)
        res_fit['bic'].append(rl.n_params*np.log(sim_data[s].shape[0]) + 2*res_opt.fun)

    print(1)









# res = minimize(fun, x0, args=(a,), method='L-BFGS-B')
# print(res)