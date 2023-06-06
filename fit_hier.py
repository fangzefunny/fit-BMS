import numpy as np
import pandas as pd 
import warnings
import pickle
import pingouin as pg 

from scipy.special import log_softmax, softmax, psi, gammaln
from scipy.stats import norm, gamma, beta
from scipy.optimize import minimize

from functools import partial

import matplotlib.pyplot as plt 
import seaborn as sns 

'''
@ Zeming Fnag 
'''

eps_ = 1e-16

# ---------------------------#
#          Colors            #
#----------------------------#

r1 = np.array([199, 111, 132]) / 255
r2 = np.array([235, 179, 169]) / 255
RedPairs  = [r1, r2]
    
b1 = np.array([ 14, 107, 168]) / 255
b2 = np.array([166, 225, 250]) / 255
BluePairs = [b1, b2]

sns.set_context('talk')
sns.set_style("ticks", {'axes.grid': False})

# ---------------------------#
#        Fit hierarchy       #
#----------------------------#

def fit_hier(data, model, nStart=5, seed=2023, tol=1e-4, max_iter=10):
    '''Hierarchical model fitting, searching for prior

    ----------------------------------------------------------------
    REFERENCES:
    
    Huys, Q. J., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, 
    R. J., & Dayan, P. (2011). Disentangling the roles of approach, 
    activation and valence in instrumental and pavlovian responding. 
    PLoS computational biology, 7(4), e1002028.
    -------------------------------------------------------------------
    Based on: https://github.com/sjgershm/mfit

    @ ZF
    '''
    # number of parameter, and possible bound
    n_param = model.n_param
    m_data  = data[list(data.keys())[0]][0].shape[0]
    n_sub   = len(data.keys())
    plb = np.array([b[0] for b in model.pbnds])
    pub = np.array([b[1] for b in model.pbnds])

    # init group-level parameters
    mus = plb + .5*(pub-plb)
    vs  = pub-plb

    # run EM until converge
    epi = 0 
    lme = 0
    while True:
        epi += 1
        prev_lme = lme 
        print(f'\nGroup-level Iteration: {epi}')

        # construct prior
        logpr = lambda x, mu, sig, link: norm(mu, sig).logpdf(link(x))
        model.logpriors = [partial(logpr, mu=mu, sig=np.sqrt(v), link=link)
                           for mu, v, link in zip(mus, vs, model.link_fns)]
        
        # E-step: optimize individual parameters
        fit_info = fit_MAP(data, model, nStart=nStart, seed=seed+1)
        
        # transform the parameter to Gaussian space,
        # using link function 
        params, params_orig = [], []
        for _, item in fit_info.items():
            params_orig.append(item['param'])
            param = [model.link_fns[i](item['param'][i]) for i in range(n_param)]
            params.append(param)
        params = np.vstack(params) # n_sub x n_param
        params_orig = np.vstack(params_orig)

        # M-step: update group-level parameters 
        # u = 1/N \sum_i m_i
        mus  = np.mean(params, axis=0)
        # v2 = 1/N \sum_i [m_i^2 + ∑^2] - mu^2
        vs = 0
        group_ll, good_h = [], []
        for i, (_, item) in enumerate(fit_info.items()):
            vs += (params[i, :])**2 + np.diag(item['H_inv'])
            try:
                log_h = np.linalg.slogdet(item['H'])[1]
                l = item['log_post'] + .5*(n_param*np.log(2*np.pi) - log_h)
                gh = 1
            except:
                warnings.warn('Hessian could not be calculated')
                l = np.nan 
                gh = 0
                continue
            group_ll.append(l)
            good_h.append(gh)
        # make sure the variance is not to small 
        vs = np.clip(vs/n_sub - mus**2, a_min=1e-5, a_max=np.inf)
        lme = np.sum(group_ll)-n_param*np.log(m_data*n_sub)

        # disp
        print(f'Finish {epi}-th iteration: \tThe group LME is {lme:.3f}')
        print(np.round(params_orig*np.array(model.scales), 4))

        # check convergence
        done = (np.abs(lme - prev_lme) < tol) or (epi >= max_iter)
        if done: 
            fit_info['group_lme'] = lme 
            fit_info['group_mu']  = mus 
            fit_info['group_v']   = vs
            break 
        
    return fit_info

# ------------------------------#
#            Fit MAP            #
#-------------------------------#

def fit_MAP(data, model, nStart=5, seed=2022):
    '''Fit model with MAP
    '''
    sub_lst = list(data.keys())

    sub_fit_res = {}
    for s in sub_lst:
        fit_info = {}

        # optimiz with multi start pts, 
        # and choose the lowest loss 
        subj_data = data[s]
        results = [model.fit(subj_data[0], seed+i) for i in range(nStart)]
        idx = np.argmin([res.fun for res in results])
        res_opt = results[idx]
        log_like = -model._negloglike(res_opt.x, subj_data[0])

        # log the fit results
        mlog = np.log(subj_data[0].shape[0])
        fit_info['log_post'] = -res_opt.fun
        fit_info['log_like'] = -log_like
        fit_info['param']    = res_opt.x
        fit_info['n_param']  = rl.n_param
        fit_info['aic']      = 2*rl.n_param-2*log_like # 2K - 2LLH  
        fit_info['bic']      = mlog*rl.n_param-2*log_like # K*log(N) - 2LLH 
        fit_info['H']        = np.linalg.pinv(res_opt.hess_inv.todense())
        fit_info['H_inv']    = res_opt.hess_inv.todense()
    
        sub_fit_res[s] = fit_info 

    return sub_fit_res

# ------------------------------#
#      Simulated Experiment     #
#-------------------------------#

class rl:
    name      = 'model 1'
    logpriors = [lambda x: gamma(a=1, scale=5).logpdf(x), 
                 lambda x: beta(a=1.2, b=1.2).logpdf(x)]
    link_fns  = [lambda y: np.log(y+eps_), 
                 lambda y: np.log(y+eps_) - np.log(1-y+eps_)]
    bnds      = [(0, 1), (0, 1)]
    pbnds     = [(0,.1), (0,.5)]
    scales    = [50, 1]
    n_param   = len(bnds)

    def __init__(self, nA):
        self.nA   = nA
        
    def sim(self, params, T=100, R=[.2, .8]):
 
        self.v = np.zeros([self.nA,])
        data = {'act': [], 'rew': [], 'trial': []}

        # decompose parameters
        self.beta  = params[0]
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
        tot_loss = self._negloglike(params, data) \
                 + self._neglogprior(params)
        return tot_loss
            
    def _negloglike(self, params, data):

        self.v = np.zeros([self.nA,])
        nll = 0 

        # decompose, reparameterize parameters
        # to ensure the eigen-value of the hessian
        # are all around 1.
        self.beta  = self.scales[0]*params[0]
        self.alpha = self.scales[1]*params[1]

        for _, row in data.iterrows():

            act = row['act']
            rew = row['rew']

            log_p = log_softmax(self.beta*self.v)
            nll -= log_p[act]
            rpe = (rew-self.v[act])
            self.v[act] += self.alpha*rpe

        return nll 

    def _neglogprior(self, params):
        logprior = 0
        for i, param in enumerate(params):
            logpr = self.logpriors[i]
            logprior += -np.max([logpr(param), -1e12]) 
        return logprior

#--------- get data  -------- #

def get_data(model, nSub=4):
    params = [[8, 0.1], [6, 0.2], [2, 0.9], [5, 0.3], [.2, .2], [3, .5], [11, .01]]
    sim_data = {}

    for s in range(nSub):
        sim_data[s] = {0: model.sim(params[s])}

    return sim_data 

def show_param():

    scales = np.array(rl(2).scales)
    true = np.vstack([[8, 0.1], [6, 0.2], [2, 0.9], [5, 0.3], [.2, .2], [3, .5], [11, .01]])

    m = 'model 1'
    with open(f'data/fit_info_{m}-hier.pkl', 'rb')as handle:
        fit_info = pickle.load(handle)
    params1 = np.vstack([fit_info[k]['param']*scales for k in range(len(true))])
    
    m = 'model 1'
    with open(f'data/fit_info_{m}-map.pkl', 'rb')as handle:
        fit_info = pickle.load(handle)
    params2 = np.vstack([fit_info[k]['param']*scales for k in range(len(true))])

    print(f'hier: \n{params1}')
    print(f'map: \n{params2}')

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    p_name = ['beta', 'alpha']
    xmax = [15, 1]
    for i in range(2): 
        ax = axs[i]
        sns.scatterplot(x=true[:, i], y=params1[:, i], color=r1, ax=ax, label='hier')
        sns.scatterplot(x=true[:, i], y=params2[:, i], color=b1, ax=ax, label='map')
        ax.legend()
        #pg.corr(x=true[:, i], y=params1[:, i])
        ax.plot(np.linspace(0, xmax[i], 20), np.linspace(0, xmax[i], 20),
                         color='k', ls='--', lw=1)
        ax.set_ylabel('Truth')
        ax.set_xlabel('Est.')
        ax.set_title(f'{p_name[i]}')
        ax.set_box_aspect(1)
    fig.tight_layout()
    plt.savefig('HIER v.s. MAP.png', dpi=300)


if __name__ == '__main__':

    # get data 
    nStart, nSub, nA = 10, 7, 2
    seed = 21242
    np.random.seed(seed)
    sim_data = get_data(rl(nA), nSub=nSub)

    # fit hierarchical
    models = [rl(nA)]
    for model in models:
        fit_info = fit_hier(sim_data, model, nStart=nStart, seed=seed)
        with open(f'data/fit_info_{model.name}-hier.pkl', 'wb')as handle:
            pickle.dump(fit_info, handle)

    # fit map 
    models = [rl(nA)]
    for model in models:
        fit_info = fit_MAP(sim_data, model, nStart=nStart, seed=seed)
        with open(f'data/fit_info_{model.name}-map.pkl', 'wb')as handle:
            pickle.dump(fit_info, handle)

    show_param()
    
    






