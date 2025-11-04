"""
Preamble for most code and jupyter notebooks
@author: bridgetsmart
@notebook date: 28 Apr 2023
"""

import numpy as np
import scipy
import scipy.integrate as integrate
import bisect
from numpy.random import default_rng


# using Ogata's thinning algorithm

# see p52 https://pat-laub.github.io/pdfs/honours_thesis.pdf for pesudocode

def conditional_intensity(t,alpha,beta,events):
    # filter events to be only < time
    e_use = [x for x in events if x<=t]
    if len(e_use)==0:
        return 0
    else:
        # sum over all conditional intensities
        return alpha * sum(np.exp(-beta*(t-e_use[i])) for i in range(len(e_use)))

def get_function_max(f,l,u):
    d = lambda x: -f(x)
    try:
        opt_res = -scipy.optimize.brent(d,brack=(l,u), full_output=True)[1]
    except:
        opt_res = -100
    return np.nanmax([opt_res, f(l),f(u)]) # just check edges

    
def get_function_inverse_closest(f,l,u):
    # returns a min, max tuple
    d = lambda x: -f(x)
    return -d(scipy.optimize.fminbound(d,l,u))


# generating times from a poisson process
def gen_poisson_homogeneous_single(l):
    """
    Generate an exponential a time from a homogeneous Poisson process with rate l.
    """
    u = np.random.uniform()
    return -np.log(u)/l


def gen_poisson_ih(l,T, troubleshooting=False):
    """
    Generate a set of times between 0 and T using a time inhomogeneous Poisson process with rate given by the function l.
    Algorithm outlined https://web.ics.purdue.edu/~pasupath/PAPERS/2011pasB.pdf
    (algorithm 6)
    """
    # we have a max time
    times = []
    t = 0

    l_capped = lambda x : np.max(l(x),0) #jic anything goes negative
    max_l = get_function_max(l_capped,0,T)

    if troubleshooting:
        pr_s = []
        
    while t < T:
        u = np.random.uniform()
        
        proposed = t - np.log(u) / max_l
        pr = l_capped(proposed) / max_l
        # print(f'current time {t}, proposed time {proposed}, max_l {max_l}, pr {pr}\n')
        if troubleshooting:
            pr_s.append([proposed, t, max_l, pr])

        t=proposed

        if (np.random.rand() <= pr): # accept
            if proposed < T:
                times.append(t)
            else:
                break

    if troubleshooting:
        return np.array(times), np.array(pr_s)
    
    return np.array(times)

def sim_hawkes(t,T,alpha,beta,events, baseline_intensity, plot=False, dt=0.1):
    if plot:
        intensity = []
        times = []
        M_save = []
        n=0

    while t<T:
        # start with first event
        M = get_function_max(baseline_intensity,t,T)+conditional_intensity(t,alpha,beta,events)
        proposed = t+gen_poisson_homogeneous_single(M) # always accept first event
        u = M*np.random.rand()

        if (u < (baseline_intensity(proposed)+conditional_intensity(proposed,alpha,beta,events))):
            if (proposed<T):
                if plot:
                    for del_time in np.arange(t,proposed,dt):
                        intensity.append(baseline_intensity(del_time)+conditional_intensity(del_time,alpha,beta,events))
                        times.append(del_time)
                        M_save.append(M)

                events.append(proposed)
                t = proposed
            else:
                # print('Reached end of time')
                if plot:
                    for del_time in np.arange(t,T,dt):
                        intensity.append(baseline_intensity(del_time)+conditional_intensity(del_time,alpha,beta,events))
                        times.append(del_time)
                        M_save.append(M)
                t = T
    
    events.sort()

    if plot:
        return events, M_save, intensity, times
    return events


# generating times from a poisson process
def gen_poisson_set(l, T):
    """
    Generate a set of times between 0 and T using a time homogeneous Poisson process with rate lambda_.
    """
    times = []
    t = 0
    while t < T:
        u = np.random.uniform()
        t += -np.log(u)/l
        if t < T:
            times.append(t)
    return np.array(times)


from collections.abc import Iterable

# need a function to flatten irregular list of lists
def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x