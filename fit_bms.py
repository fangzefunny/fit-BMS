import numpy as np
import pandas as pd 

from scipy.special import log_softmax, softmax, psi, gammaln
from scipy.stats import beta, gamma
from scipy.optimize import minimize

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
#      Group level BMS       #
#----------------------------#

# ------ fit BMS ---- #

def fit_bms(fit_results, tol=1e-4):
    '''Fit group-level Bayesian model seletion
    Nm is the number of model

    Args: 
        fit_results: [Nm, list] a list of model fitting results
    
    Outputs:
        BMS result: a dict including 
            -alpha: [1, Nm] posterior of the model probability
            -p_m1D: [nSub, Nm] posterior of the model 
                     assigned to the subject data p(m|D)
            -E_r1D: [nSub, Nm] expectation of E[p(r|D)]
            -xp:    [Nm,] exceedance probabilities
            -bor:   [1] Bayesian Omnibus Risk, the probability
                    of choosing null hypothesis: model frequencies are equal
            -pxp:   [Nm,] protected exceedance probabilities

    ----------------------------------------------------------------
    REFERENCES:
    
    Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
    
    Rigoux, L, Stephan, KE, Friston, KJ and Daunizeau, J. (2014)
    Bayesian model selection for group studiesRevisited.
    NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
    -------------------------------------------------------------------
    Based on: https://github.com/sjgershm/mfit

    @ ZF
    '''
    
    ## get log model evidence
    lme = np.vstack([calc_lme(model_res) 
                for model_res in fit_results]).T
    
    ## get group-level posterior
    Nm = lme.shape[1]
    alpha0, alpha = np.ones([1, Nm]), np.ones([1, Nm])

    while True:
        
        # cache previous α
        prev = alpha.copy()

        # compute the posterior: Nsub x Nm
        # p(m|D) (p, k) = exp[log p(D(p,1))|m(p,k)) + Psi(α(1,k)) - Psi(α'(1,1))]
        log_u = lme + psi(alpha) - psi(alpha.sum())
        u = np.exp(log_u - log_u.max(1, keepdims=True)) # the max trick 
        p_m1D = u / u.sum(1, keepdims=True)

        # compute beta: 1 x Nm
        # β(k) = sum_p p(m|D)
        B = p_m1D.sum(0, keepdims=True)

        # update alpha: 1 x Nm
        # α(k) = α0(k) + β(k) 
        alpha = alpha0 + B 

        # check convergence 
        if np.linalg.norm(alpha - prev) < tol:
            break 
    
    # get the expected posterior 
    E_r1D = alpha / alpha.sum()

    # get the exeedence probabilities 
    xp = dirchlet_exceedence(alpha)

    # get the Bayesian Omnibus risk
    bor = calc_BOR(lme, p_m1D, alpha, alpha0)
    
    # get the protected exeedence probabilities
    pxp=(1-bor)*xp+bor/Nm

    # out BMS fit 
    BMS_result = { 'alpha_post': alpha, 'p_m1D': p_m1D, 
                   'E_r1D': E_r1D, 'xp': xp, 'bor': bor, 'pxp': pxp}

    return BMS_result

def calc_lme(model_res):
    '''Calculate Log Model Evidence

    Turn a list of fitting results of different
    model into a matirx lme. Ns means number of subjects, 
    Nm is the number of models.

    Args:
        model_res: [dict,] A dict of model's fitting info
            - log_post: opt parameters
            - log_like: log likelihood
            - param: the optimal parameters
            - n_param: the number of parameters
            - aic
            - bic
            - H: hessian matrix 
    
    Outputs:
        lme: [Ns, Nm] log model evidence 
                
    '''
    lme  = []
    for s in range(len(model_res['log_post'])):
        # log|-H|
        h = np.log(np.linalg.det(model_res['H'][s]))
        # log p(D,θ*|m) + .5(log(d) - log|-H|) 
        l = model_res['log_post'][s] + \
            .5*(model_res['n_param']*np.log(2*np.pi)-h)
        lme.append(l)
    # use BIC if any Hessians are degenerate 
    ind = np.isnan(lme) | np.isinf(lme)| (np.imag(lme)!=0)
    if any(ind.reshape([-1])): lme = -.5 * model_res['bic']
            
    return np.array(lme)

def dirchlet_exceedence(alpha_post, nSample=1e6):
    '''Sampling to calculate exceedence probability

    Args:
        alpha: [1,Nm] dirchilet distribution parameters
        nSample: number of samples

    Output: 
    '''
    # the number of categories
    Nm = alpha_post.shape[1]
    alpha_post = alpha_post.reshape([-1])

    # sampling in blocks
    blk = int(np.ceil(nSample*Nm*8 / 2**28))
    blk = np.floor(nSample/blk * np.ones([blk,]))
    blk[-1] = nSample - (blk[:-1]).sum()
    blk = blk.astype(int)

    # sampling 
    xp = np.zeros([Nm,])
    for i in range(len(blk)):

        # sample from a gamma distribution and normalized
        r = np.vstack([gamma(a).rvs(size=blk[i]) for a in alpha_post]).T
        r = r / r.sum(1, keepdims=True)

        # use the max decision rule and count 
        xp += np.bincount(np.argmax(r, axis=1))

    return xp / nSample

# -------- Bayesian Omnibus Risk -------- #

def calc_BOR(lme, p_m1D, alpha_post, alpha0):
    '''Calculate the Bayesian Omnibus Risk

     Args:
        lme: [Nsub, Nm] log model evidence
        p_r1D: [Nsub, Nm] the posterior of each model 
                        assigned to the data
        alpha_post:  [1, Nm] H1: alpha posterior 
        alpha0: [1, Nm] H0: alpha=[1,1,1...]

    Outputs:
        bor: the probability of selection the null
                hypothesis.
    '''
    # calculte F0 and F1
    f0 = F0(lme)
    f1 = FE(lme, p_m1D, alpha_post, alpha0)

    # BOR = 1/(1+exp(F1-F0))
    bor = 1 / (1+ np.exp(f1-f0))
    return bor 

def F0(lme):
    '''Calculate the negative free energy of H0

    Args:
        lme: [Nsub, Nm] log model evidence

    Outputs:
        f0: negative free energy as an approximation
            of log p(D|H0)
    '''
    Nm = lme.shape[1]
    qm = softmax(lme, axis=1)    
    f0 = (qm * (lme - np.log(Nm) - np.log(qm + eps_))).sum()                                  
    return f0
    
def FE(lme, p_m1D, alpha_post, alpha0):
    '''Calculate the negative free energy of H1

    Args:
        lme: [Nsub, Nm] log model evidence
        p_m1D: [Nsub, Nm] the posterior of each model 
                        assigned to the data
        alpha_post:  [1, Nm] H1: alpha posterior 
        alpha0: [1, Nm] H0: alpha=[1,1,1...]

    Outputs:
        f1: negative free energy as an approximation
            of log p(D|H1)
    '''
    E_log_r = psi(alpha_post) - psi(alpha_post.sum())
    E_log_rmD = (p_m1D*(lme+E_log_r)).sum() + ((alpha0 -1)*E_log_r).sum()\
                + gammaln(alpha0.sum()) - (gammaln(alpha0)).sum()
    Ent_p_r1D = -(p_m1D*np.log(p_m1D + eps_)).sum()
    Ent_alpha  = gammaln(alpha_post).sum() - gammaln(alpha_post.sum()) \
                                        - ((alpha_post-1)*E_log_r).sum()
    f1 = E_log_rmD + Ent_p_r1D + Ent_alpha
    return f1

# ------------------------------#
#            Fit MAP            #
#-------------------------------#

def fit_MAP(data, models, nStart=5, seed=2022):
    '''Fit model with MAP
    '''
    nSub = len(data.keys())
    fields = ['log_post', 'log_like', 'param', 'n_param', 'aic', 'bic', 'H']

    fit_results = []
    for model in models:
        model_res = {k: [] for k in fields}

        for s in range(nSub):
            # optimiz with multi start pts, 
            # and choose the lowest loss 
            subj_data = data[s]
            results = [model.fit(subj_data, seed+i) for i in range(nStart)]
            idx = np.argmin([res.fun for res in results])
            res_opt = results[idx]
            log_like = -model._negloglike(res_opt.x, subj_data)

            # log the fit results
            model_res['log_post'].append(-res_opt.fun)
            model_res['log_like'].append(log_like)
            model_res['param'].append(res_opt.x)
            model_res['n_param'] = rl.n_param
            model_res['aic'].append(2*rl.n_param - 2*log_like) # 2K - 2LLH  
            model_res['bic'].append(rl.n_param*np.log(subj_data.shape[0]) 
                                    - 2*log_like) # # K*log(N) - 2LLH 
            model_res['H'].append(np.linalg.inv(res_opt.hess_inv.todense()))
        
        fit_results.append(model_res)

    return fit_results

# ------------------------------#
#      Simulated Experiment     #
#-------------------------------#

class rl:
    name    = 'model 1'
    priors  = [gamma(a=1, scale=5), beta(a=1.2, b=1.2)]
    bnds    = [(0, 50), (0, 1)]
    pbnds   = [(0, 10), (0,.5)]
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
        tot_loss = self._negloglike(params, data) \
                 + self._neglogprior(params)
        return tot_loss
                    
    def _negloglike(self, params, data):

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
            rpe = (rew-self.v[act])
            self.v[act] += self.alpha*rpe

        return nll 

    def _neglogprior(self, params):
        logprior = 0
        for prior, param in zip(self.priors, params):
            logprior += -np.max([prior.logpdf(param), -1e12])
        return logprior

class rl2(rl):
    name    = 'model 2'
    priors  = [gamma(a=1, scale=5), beta(a=1.2, b=1.2), beta(a=1.2, b=1.2)]
    bnds    = [(0, 50), (0, 1), (0, 1)]
    pbnds   = [(0, 10), (0,.5), (0,.5)]
    n_param = len(bnds)

    def __init__(self, nA):
        super().__init__(nA)

    def _negloglike(self, params, data):

        self.v = np.zeros([self.nA,])
        nll = 0 

        # decompose parameters
        self.beta = params[0]
        self.alpha_pos = params[1]
        self.alpha_neg = params[2]

        for _, row in data.iterrows():

            act = row['act']
            rew = row['rew']

            log_p = log_softmax(self.beta*self.v)
            nll -= log_p[act]
            rpe = rew-self.v[act]
            alpha = self.alpha_neg if (rpe<=0) else self.alpha_pos
            self.v[act] += alpha*rpe

        return nll 

#--------- get data  -------- #

def get_data(model, nSub=4):
    params = [[8, 0.1], [6, 0.2], [2, 0.1], [5, 0.3]]
    sim_data = {}

    for s in range(nSub):
        sim_data[s] = model.sim(params[s])

    return sim_data 

if __name__ == '__main__':

    # get data 
    nStart, nSub, nA = 2, 4, 2
    seed = 2123
    sim_data = get_data(rl(nA), nSub=nSub)

    # models 
    models = [rl(nA), rl2(nA)]
    
    # fit MAP
    fit_res = fit_MAP(sim_data, models, nStart=5, seed=seed)
    avg_log_like = np.array([np.mean(r['log_post']) for r in fit_res])

    # fit group-level BMS
    bms_res = fit_bms(fit_res)

    # compare avg NLL and BMS
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    sns.barplot(x=[0,1], y=-avg_log_like, palette=RedPairs, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Model 1', 'Model 2'])
    ax.set_xlim([-.8, 1.8])
    ax.set_ylabel('Avg. NLL')
    ax = axs[1]
    sns.barplot(x=[0,1], y=bms_res['pxp'], palette=BluePairs, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Model 1', 'Model 2'])
    ax.set_xlim([-.8, 1.8])
    ax.set_ylabel('PXP')
    fig.tight_layout()
    plt.savefig('NLL v.s. PXP.png', dpi=300)
    






