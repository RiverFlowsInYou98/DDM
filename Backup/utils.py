import numpy as np
from scipy.special import logsumexp

def MCsample(P, init_dist, k): 
    """
    sample a 'k'-step Markov chain 
    with initial distribution 'init_dist'
    and transition probabilities 'P'
    """
    s = len(init_dist) # num of states
    X = np.zeros(k)
    X[0] = np.random.choice(s, p=init_dist)
    for i in range(1, k):
        X[i] = np.random.choice(s, p=P[int(X[i-1]), :])
    return X

def logdotexp(A, B):
    """
    numerically stable ways to compute
    np.log(np.dot(np.exp(A), np.exp(B)))
    A, B are 2d arrays satisfying A.shape[1] = B.shape[0]
    """
    assert len(A.shape) == 2
    assert len(B.shape) == 2
    assert A.shape[1] == B.shape[0]
    A_ = np.stack([A] * B.shape[1]).transpose(1, 0, 2)
    B_ = np.stack([B] * A.shape[0]).transpose(0, 2, 1)
    return logsumexp(A_ + B_, axis=2)

def loss_fun(chain, data):
    """
    compute the negative log likelihood
    likelihood = product of P(X(Tk)=Ck)
    """
    logprob = 0 
    for Tk, Ck in data:
        logprob -= np.log(chain.ExitProb_logdp2(Tk, Ck))
        if np.isinf(logprob):
            raise ValueError('Infty detected, computation is stopped.')
    return logprob / len(data)

def wfpt(T, mu, a, z, err):
    tt = T / (a ** 2)  # use normalized time
    w = z / a  # convert to relative start point
    # calculate number of terms needed for large T
    if np.pi * tt * err < 1:  # if error threshold is set low enough
        kl = np.sqrt(-2 * np.log(np.pi * tt * err) / (np.pi ** 2 * tt))  # bound
        kl = max(kl, 1 / (np.pi * np.sqrt(tt)))  # ensure boundary conditions met
    else:  # if error threshold set too high
        kl = 1 / (np.pi * np.sqrt(tt))  # set to boundary condition
    # calculate number of terms needed for small T
    if 2 * np.sqrt(2 * np.pi * tt) * err < 1:  # if error threshold is set low enough
        ks = 2 + np.sqrt(-2 * tt * np.log(2 * np.sqrt(2 * np.pi * tt) * err))  # bound
        ks = max(ks, np.sqrt(tt) + 1)  # ensure boundary conditions are met
    else:  # if error threshold was set too high
        ks = 2  # minimal kappa for that case
    # compute f(tt|0,1,w)
    p = 0
    if ks < kl:  # if small T is better...
        K = np.ceil(ks)
        for k in range(-int(np.floor((K - 1) / 2)), int(np.ceil((K - 1) / 2)) + 1):  # loop over k
            p = p + (w + 2 * k) * np.exp(-((w + 2 * k) ** 2) / 2 / tt)
        p = p / np.sqrt(2 * np.pi * tt ** 3)
    else:  # if large T is better...
        K = np.ceil(kl)
        for k in range(1, int(K) + 1):
            p = p + k * np.exp(-(k ** 2) * (np.pi ** 2) * tt / 2) * np.sin(k * np.pi * w)
        p = p * np.pi
    # convert to f(T|mu,a,w)
    p = p * np.exp(-mu * a * w - (mu ** 2) * T / 2) / (a ** 2)
    return p