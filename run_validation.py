""" 
Preamble for most code and jupyter notebooks 
@author: bridgetsmart 

@notebook date: 26th Nov 2024
"""  
from collections import defaultdict
import numpy as np, pandas as pd  
import matplotlib.pyplot as plt, seaborn as sns 
import matplotlib as mpl  
import math, string, re, pickle, json, time, os, sys, datetime, itertools  
from tqdm.notebook import tqdm


from functions.cross_correlogram import *
from functions.simulating_hawkes import *

# circadian / time of day effect

from scipy.optimize import curve_fit

import scipy.integrate as integrate
from functions.significance_testing_MTC import multiple_test_poisson, scale_for_fraction



from collections.abc import Iterable

# need a function to flatten irregular list of lists
def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

from scipy.stats import chisquare
import scipy.integrate as integrate

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


from cross_correlogram import *
from simulating_hawkes import *

from sinusoidal_jitter_error import get_plots_periodic, jitter_times, bursty,  get_plots_bursty, inactive_bursty, sim_ct


def granger_min_pvalue_from_events(times_x, times_y, bin_width,
                                   maxlag=5, alpha=0.05,
                                   align_method='intersection',
                                   verbose=False):
    """
    Run a Granger causality test on two event-time series (converted to binned counts)
    and return the most significant p-value across all tested lags.

    Parameters
    ----------
    times_x, times_y : 1D numpy arrays
        Event times (floats).
    bin_width : float
        Width of time bin (same units as event times).
    maxlag : int, default 5
        Maximum lag (in bins) to test.
    alpha : float, default 0.05
        Significance level (used only for labeling result).
    align_method : {'intersection', 'union'}, default 'intersection'
        Whether to align time ranges by intersection or union.
    verbose : bool, default False
        Print detailed statsmodels output.

    Returns
    -------
    p_min : float
        Smallest p-value found among tested lags.
    lag_best : int
        Lag (in bins) corresponding to smallest p-value.
    is_significant : bool
        Whether p_min < alpha.
    """
    tx = np.asarray(times_x)
    ty = np.asarray(times_y)
    if tx.size == 0 or ty.size == 0:
        raise ValueError("Both event arrays must be non-empty.")

    # Determine time window
    min_x, max_x = tx.min(), tx.max()
    min_y, max_y = ty.min(), ty.max()

    if align_method == 'intersection':
        t_start = max(min_x, min_y)
        t_end = min(max_x, max_y)
        if t_end <= t_start:
            raise ValueError("No overlap between event time ranges for 'intersection'.")
    else:  # union
        t_start = min(min_x, min_y)
        t_end = max(max_x, max_y)

    # Create bins
    bins = np.arange(t_start, t_end + bin_width, bin_width)
    if len(bins) < 2:
        raise ValueError("bin_width too large for this time range.")

    # Histogram to counts per bin
    counts_x, _ = np.histogram(tx, bins=bins)
    counts_y, _ = np.histogram(ty, bins=bins)
    df = pd.DataFrame({'x': counts_x, 'y': counts_y}, index=bins[:-1])

    # Run Granger test: does x -> y ?
    data = np.column_stack([df['y'].values, df['x'].values])
    gc_res = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)

    # Collect p-values
    pvals = []
    for lag in range(1, maxlag + 1):
        try:
            p = gc_res[lag][0]['ssr_ftest'][1]
        except Exception:
            p = np.nan
        pvals.append((lag, p))

    # Find most significant lag
    pvals = [(lag, p) for lag, p in pvals if np.isfinite(p)]
    if not pvals:
        raise RuntimeError("No valid p-values returned.")
    lag_best, p_min = min(pvals, key=lambda t: t[1])
    return p_min

def expected_target_events_hj_delta(si, hj, delta, lambda_T):
    l= si + hj
    return integrate.quad(lambda_T, l, l+delta)[0] # integral from si+hj to si+hj+delta of lambda_T
def expected_bin_height(hj, lambda_S, lambda_T, delta, w, T):
    term = lambda x : lambda_S(x) * expected_target_events_hj_delta(x, hj, delta, lambda_T)
    return integrate.quad(term, w, T-w)[0] # integral from si+hj to si+hj+delta of lambda_S * E[events from T]
def sd_bin_height(hj, lambda_S, lambda_T, delta, w, T):
    term = lambda x : lambda_S(x) * expected_target_events_hj_delta(x, hj, delta, lambda_T)
    return integrate.quad(term, w, T-w)[0] # integral from si+hj to si+hj+delta of lambda_S * E[events from T]

def get_deltas(t1, t2, win, T):
    deltas = []
    for t in np.array_split(t1, 6): # auspol occurs 3.6 times more than the second most frequent - this is a safe value
        deltas.append(possible_time_del(t, t2, win, T))

    deltas = np.array(list(flatten(deltas)))

    return deltas
def validate_chisquare(obs_pos, exp_pos):
    return chisquare(f_obs=obs_pos, f_exp=exp_pos*np.sum(obs_pos)/np.sum(exp_pos))[1]
def func(x,a,b,c):
    return 1*(np.sin((x-b)/(24/(2*np.pi)))+1.07)

def func_with_pen(x,a,b,c):
    return func(x, a, b,c) + 1e8*(func(0, a,b,c)-func(24, a, b,c))

def get_tod_bin_heights(df, delta, win, T, tag_A, tag_B):


    def lambda_s_tod(x):
        return 1*(np.sin((x-3)/(24/(2*np.pi)))+1.07) #+ 1/30*(np.sin((x-4)/(30*60*24/(2*np.pi))))+0.02  # period of one day

    def lambda_t_tod(x):
        return 1*(np.sin((x-6)/(24/(2*np.pi)))+1.07)#+ 1/20*(np.sin((x-14)/(30*60*24/(2*np.pi))))+0.02  # period of one day out by a bit



    # x=np.arange(0,24,1)

    # # no spike
    # y1_no=df[df.tag==tag_A].groupby('hour').count()['tag'].reindex(index=x, fill_value=0).values
    # y_min_no = np.min(y1_no)
    # fit_p1_no,_ = curve_fit(func_with_pen, x, y1_no/(T//24))

    # y2_no=df[df.tag==tag_B].groupby('hour').count()['tag'].reindex(index=x, fill_value=0).values
    # y_min2_no = np.min(y2_no)
    # fit_p2_no,_ = curve_fit(func_with_pen, x, y2_no/(T//24))

    # def lambda_s(x):
    #     # divide by number of days in dataset
    #     x = np.mod(x,24)
    #     return  func_with_pen(x, *fit_p1_no)

    # def lambda_t(x):
    #     # divide by number of days in dataset
    #     x = np.mod(x,24)
    #     return func_with_pen(x, *fit_p2_no)

    # b_heights = {}
    if not os.path.exists(f'tod_bins_{tag_A}_{tag_B}.pkl'):

        tod_bins = 2*np.array([expected_bin_height(x, lambda_s_tod, lambda_t_tod, delta, win, T) for x in left_bin_edges(win, delta)])
        # store tod_bins
        with open(f'tod_bins_{tag_A}_{tag_B}.pkl', 'wb') as f:
            pickle.dump(tod_bins, f)

    else:
        with open(f'tod_bins_{tag_A}_{tag_B}.pkl', 'rb') as f:
            tod_bins = pickle.load(f)

    return tod_bins


def safe_jitter(times, jitter_window, S, T):
    j = jitter_times(times, jitter_window, S, T)
    j = np.clip(j, S, T)
    return np.sort(j)

# get the average rate for each interval and return as a np array
def get_rates(t, interval_size, t_min, t_max):
    intervals = np.arange(t_min, t_max, interval_size)
    rates = []
    for i in range(len(intervals)-1):
        rates.append(len(t[(t>=intervals[i]) & (t<intervals[i+1])]) / interval_size)
    return intervals[:-1], np.array(rates)
def get_smooth_bin_heights(t1, t2, delta, win, T, interval_size):
    s_times = t1
    t_times = t2

    # helper that looks up rate by index
    def lambda_func(x, i__, r__):
        if x < win:
            return 0
        if x > (T-win):
            return 0
        else:
            # find first index where interval >= x
            idx = np.searchsorted(i__, x, side='right') - 1
            if idx < 0 or idx >= len(r__):
                return 0
            return r__[idx]
            

    i_s, r_s = get_rates(s_times, interval_size, win,np.max([np.max(s_times),T-win])+2*interval_size)
    i_t, r_t = get_rates(t_times, interval_size, win, np.max([np.max(t_times),T-win])+2*interval_size)

    lambda_s = lambda x : lambda_func(x, i_s, r_s)
    lambda_t = lambda x : lambda_func(x, i_t, r_t)
    return [expected_bin_height(x, lambda_s,  lambda_t, delta, win, T) for x in left_bin_edges(win, delta)]
def lambda_s(x):
    return 1*(np.sin((x-3)/(24/(2*np.pi)))+1.07) #+ 1/30*(np.sin((x-4)/(30*60*24/(2*np.pi))))+0.02  # period of one day

def lambda_t(x):
    return 1*(np.sin((x-6)/(24/(2*np.pi)))+1.07)#+ 1/20*(np.sin((x-14)/(30*60*24/(2*np.pi))))+0.02  # period of one day out by a bit

# start by taking initial parameters
# start by taking initial parameters

np.random.seed(8)

T = 60*24 # 1 months
T_sample = T


win = 3
delta = 0.5

c1 = 1
c2 = 1 # 1 event per hour

# uniform sampling for bursty 
int_l = 0.5 # around 30 mins
int_u = 3 # around 3 hours

BR = 0.8 # quite high

jitter_window = 0.5 # jitter_window for jittering

n_iter = 100
results = defaultdict(list)

for alpha in np.arange(0,0.5,0.05):
    for _ in range(n_iter):
        def lambda_s(x):
            return 1*(np.sin((x-3)/(24/(2*np.pi)))+1.07) #+ 1/30*(np.sin((x-4)/(30*60*24/(2*np.pi))))+0.02  # period of one day

        def lambda_t(x):
            return 1*(np.sin((x-6)/(24/(2*np.pi)))+1.07)#+ 1/20*(np.sin((x-14)/(30*60*24/(2*np.pi))))+0.02  # period of one day out by a bit


        # take average value of lambda_s and lambda_t over one day to get approx equal numbers of events
        avg_lambda_s = integrate.quad(lambda_s, 0, 24)[0]/24
        avg_lambda_t = integrate.quad(lambda_t, 0, 24)[0]/24

        const_lambda_s = lambda x: avg_lambda_s
        const_lambda_t = lambda x: avg_lambda_t

        # generate baseline t1_no, t2_no, t1_bursty, t2_bursty, t1_periodic, t2_periodic
        t1_no = gen_poisson_ih(const_lambda_s, T_sample);
        t2_no = gen_poisson_ih(const_lambda_t, T_sample);

        # periodic
        t1_periodic = gen_poisson_ih(lambda_s, T_sample);
        t2_periodic = gen_poisson_ih(lambda_t, T_sample);

        # bursty
        lambda_s_bursty = lambda t: c1
        lambda_t_bursty = lambda t: c2

        c_t =  sim_ct((50,20), T, 0, current_state = 0) # expect to spend 50 units in a independent state and then 20 in bursty

        t1_bursty = np.array(inactive_bursty(lambda_s_bursty, BR, int_l, int_u, T_sample, c_t))
        t2_bursty = np.array(inactive_bursty(lambda_t_bursty, BR, int_l, int_u, T_sample, c_t))

        ### TO ITERATE

        # CODE TO ADD add in spike to all 3 for t2, base on following code
        t2_no_plus_spike = t2_no.copy()
        t2_periodic_plus_spike = t2_periodic.copy()
        t2_bursty_plus_spike = t2_bursty.copy()

        no_add = t1_no + np.random.normal(1, 0.3, len(t1_no))
        periodic_add = t1_periodic + np.random.normal(1, 0.3, len(t1_periodic))
        bursty_add = t1_bursty + np.random.normal(1, 0.3, len(t1_bursty))

        add_no = np.random.choice(no_add, int(len(no_add)*alpha))
        add_periodic = np.random.choice(periodic_add, int(len(periodic_add)*alpha))
        add_bursty = np.random.choice(bursty_add, int(len(bursty_add)*alpha))

        t2_no_plus_spike = np.sort(np.concatenate((t2_no_plus_spike, add_no[add_no<T])))
        t2_periodic_plus_spike = np.sort(np.concatenate((t2_periodic_plus_spike, add_periodic[add_periodic<T])))
        t2_bursty_plus_spike = np.sort(np.concatenate((t2_bursty_plus_spike, add_bursty[add_bursty<T])))

        # calculate jittered timeseries
        t1_no_jittered = safe_jitter(t1_no, jitter_window, 0, T)
        t1_periodic_jittered = safe_jitter(t1_periodic, jitter_window, 0, T)
        t1_bursty_jittered = safe_jitter(t1_bursty, jitter_window, 0, T)
        t2_no_jittered = safe_jitter(t2_no_plus_spike, jitter_window, 0, T)
        t2_periodic_jittered = safe_jitter(t2_periodic_plus_spike, jitter_window, 0, T)
        t2_bursty_jittered = safe_jitter(t2_bursty_plus_spike, jitter_window, 0, T)


        # calculate the deltas
        deltas = {}
        deltas_jittered = {}

        deltas['no_spike'] = get_deltas(t1_no, t2_no, win, T)
        deltas['periodic_spike'] = get_deltas(t1_periodic, t2_periodic, win, T)
        deltas['bursty_spike'] = get_deltas(t1_bursty, t2_bursty, win, T)

        deltas_jittered['no_spike'] = get_deltas(t1_no_jittered, t2_no_jittered, win, T)
        deltas_jittered['periodic_spike'] = get_deltas(t1_periodic_jittered, t2_periodic_jittered, win, T)
        deltas_jittered['bursty_spike'] = get_deltas(t1_bursty_jittered, t2_bursty_jittered, win, T)

        # # now the additional spike
        deltas['no_spike_plus'] = np.concatenate([deltas['no_spike'], get_deltas(t1_no, t2_no_plus_spike, win, T)])
        deltas['periodic_spike_plus'] = np.concatenate([deltas['periodic_spike'], get_deltas(t1_periodic, t2_periodic_plus_spike, win, T)])
        deltas['bursty_spike_plus'] = np.concatenate([deltas['bursty_spike'], get_deltas(t1_bursty, t2_bursty_plus_spike, win, T)])

        # ## ESTIMATE CCH espected bins under (1) null model (2) TOD model

        # estimate intensity functions for the CCH
        bins = np.arange(-win,win+delta,delta)
        to_zero = (np.where(bins == -delta)[0][0], np.where(bins == 0)[0][0])

        bin_heights = {}

        ## NAIVE

        naive_lambdas = {}
        for k in deltas.keys():
            if 'no_spike' in k:
                naive_lambdas[k] = (lambda x : len(t1_no) / (T-2*win),lambda x : len(t2_no_plus_spike) / (T-2*win))
            elif 'periodic' in k:
                naive_lambdas[k] = (lambda x: len(t1_periodic) / (T-2*win), lambda x: len(t2_periodic_plus_spike) / (T-2*win) )
            elif 'bursty' in k:
                naive_lambdas[k] = (lambda x: len(t1_bursty) / (T-2*win), lambda x: len(t2_bursty_plus_spike) / (T-2*win) )

        bin_heights['naive_no_spike'] = [expected_bin_height(x, naive_lambdas['no_spike'][0],  naive_lambdas['no_spike'][1], delta, win, T) for x in left_bin_edges(win, delta)]
        bin_heights['naive_periodic'] = [expected_bin_height(x, naive_lambdas['periodic_spike'][0],  naive_lambdas['periodic_spike'][1], delta, win, T) for x in left_bin_edges(win, delta)]
        bin_heights['naive_bursty'] = [expected_bin_height(x, naive_lambdas['bursty_spike'][0],  naive_lambdas['bursty_spike'][1], delta, win, T) for x in left_bin_edges(win, delta)]

        ## TOD    
        df_no_spike = pd.DataFrame({'tag':len(t1_no)*['A']+len(t2_no_plus_spike)*['B'], 'time':list(t1_no)+list(t2_no_plus_spike)})
        df_periodic = pd.DataFrame({'tag':len(t1_periodic)*['A']+len(t2_periodic_plus_spike)*['B'], 'time':list(t1_periodic)+list(t2_periodic_plus_spike)})   
        df_bursty = pd.DataFrame({'tag':len(t1_bursty)*['A']+len(t2_bursty_plus_spike)*['B'], 'time':list(t1_bursty)+list(t2_bursty_plus_spike)}) 

        tag_A = 'A'
        tag_B = 'B'

        df_no_spike['hours'] = np.floor(df_no_spike.time)
        df_no_spike['hour'] = np.mod(df_no_spike.hours,24)
        df_periodic['hours'] = np.floor(df_periodic.time)
        df_periodic['hour'] = np.mod(df_periodic.hours,24)
        df_bursty['hours'] = np.floor(df_bursty.time)
        df_bursty['hour'] = np.mod(df_bursty.hours,24)


        bin_heights['tod_no_spike'] = get_tod_bin_heights(df_no_spike, delta, win, T, tag_A, tag_B)
        bin_heights['tod_periodic_spike'] = get_tod_bin_heights(df_periodic, delta, win, T, tag_A, tag_B)
        bin_heights['tod_bursty_spike'] = get_tod_bin_heights(df_bursty, delta, win, T, tag_A, tag_B)

        ## SMOOTH

        interval_size = 1
        bin_heights['smooth_no_spike'] = get_smooth_bin_heights(t1_no, t2_no_plus_spike, delta, win, T, interval_size)
        bin_heights['smooth_periodic_spike']= get_smooth_bin_heights(t1_periodic, t2_periodic_plus_spike, delta, win, T, interval_size)
        bin_heights['smooth_bursty_spike']= get_smooth_bin_heights(t1_bursty, t2_bursty_plus_spike, delta, win, T, interval_size)

        ## JITTER

        naive_lambdas_jitter = {}
        for k in deltas_jittered.keys():
            if 'no_spike' in k:
                naive_lambdas_jitter[k] = (lambda x : len(t1_no_jittered) / (T-2*win),lambda x : len(t2_no_jittered) / (T-2*win))
            elif 'periodic' in k:
                naive_lambdas_jitter[k] = (lambda x: len(t1_periodic_jittered) / (T-2*win), lambda x: len(t2_periodic_jittered) / (T-2*win) )
            elif 'bursty' in k:
                naive_lambdas_jitter[k] = (lambda x: len(t1_bursty_jittered) / (T-2*win), lambda x: len(t2_bursty_jittered) / (T-2*win) )

        bin_heights['naive_no_spike_jittered'] = [expected_bin_height(x, naive_lambdas_jitter['no_spike'][0],  naive_lambdas_jitter['no_spike'][1], delta, win, T) for x in left_bin_edges(win, delta)]
        bin_heights['naive_periodic_jittered'] = [expected_bin_height(x, naive_lambdas_jitter['periodic_spike'][0],  naive_lambdas_jitter['periodic_spike'][1], delta, win, T) for x in left_bin_edges(win, delta)]
        bin_heights['naive_bursty_jittered'] = [expected_bin_height(x, naive_lambdas_jitter['bursty_spike'][0],  naive_lambdas_jitter['bursty_spike'][1], delta, win, T) for x in left_bin_edges(win, delta)]


        # calculate actual bin heights for jittered / non jittered

        H_store = {}
        _, H_store['no_spike'] = plot_cc_from_deltas(deltas['no_spike_plus'], win, delta, return_H=True)
        _, H_store['periodic_spike'] = plot_cc_from_deltas(deltas['periodic_spike_plus'], win, delta, return_H=True)    
        _, H_store['bursty_spike'] =  plot_cc_from_deltas(deltas['bursty_spike_plus'], win, delta, return_H=True)


        _, H_store['no_spike_jittered'] = plot_cc_from_deltas(deltas_jittered['no_spike'], win, delta, return_H=True)
        _, H_store['periodic_spike_jittered'] = plot_cc_from_deltas(deltas_jittered['periodic_spike'], win, delta, return_H=True)    
        _, H_store['bursty_spike_jittered'] =  plot_cc_from_deltas(deltas_jittered['bursty_spike'], win, delta, return_H=True)


        results_short = {}

        # significance testing
        sig_alpha = 0.05
        match_frac = 0.75
        results_short['naive_CC_no_spike'] = 1 if multiple_test_poisson(H_store['no_spike'], scale_for_fraction(H_store['no_spike'],bin_heights['naive_no_spike'], fraction=match_frac)[0]*np.array(bin_heights['naive_no_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['naive_CC_periodic_spike'] = 1 if multiple_test_poisson(H_store['periodic_spike'], scale_for_fraction(H_store['periodic_spike'],bin_heights['naive_periodic'], fraction=match_frac)[0]*np.array(bin_heights['naive_periodic']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['naive_CC_bursty_spike'] = 1 if multiple_test_poisson(H_store['bursty_spike'], scale_for_fraction(H_store['bursty_spike'],bin_heights['naive_bursty'], fraction=match_frac)[0]*np.array(bin_heights['naive_bursty']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['tod_CC_no_spike'] = 1 if multiple_test_poisson(H_store['no_spike'], scale_for_fraction(H_store['no_spike'],bin_heights['tod_no_spike'], fraction=match_frac)[0]*np.array(bin_heights['tod_no_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['tod_CC_periodic_spike'] = 1 if multiple_test_poisson(H_store['periodic_spike'], scale_for_fraction(H_store['periodic_spike'],bin_heights['tod_periodic_spike'], fraction=match_frac)[0]*np.array(bin_heights['tod_periodic_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['tod_CC_bursty_spike'] = 1 if multiple_test_poisson(H_store['bursty_spike'], scale_for_fraction(H_store['bursty_spike'],bin_heights['tod_bursty_spike'], fraction=match_frac)[0]*np.array(bin_heights['tod_bursty_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['smooth_CC_no_spike'] = 1 if multiple_test_poisson(H_store['no_spike'], scale_for_fraction(H_store['no_spike'],bin_heights['smooth_no_spike'], fraction=match_frac)[0]*np.array(bin_heights['smooth_no_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['smooth_CC_periodic_spike'] = 1 if multiple_test_poisson(H_store['periodic_spike'], scale_for_fraction(H_store['periodic_spike'],np.array(bin_heights['smooth_periodic_spike']), fraction=match_frac)[0]*np.array(bin_heights['smooth_periodic_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['smooth_CC_bursty_spike'] = 1 if multiple_test_poisson(H_store['bursty_spike'], scale_for_fraction(H_store['bursty_spike'],np.array(bin_heights['smooth_bursty_spike']), fraction=match_frac)[0]*np.array(bin_heights['smooth_bursty_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['naive_CC_no_spike_jitter'] = 1 if multiple_test_poisson(H_store['no_spike_jittered'], scale_for_fraction(H_store['no_spike_jittered'],np.array(bin_heights['naive_no_spike']), fraction=match_frac)[0]*np.array(bin_heights['naive_no_spike']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['naive_CC_periodic_spike_jitter'] = 1 if multiple_test_poisson(H_store['periodic_spike_jittered'], scale_for_fraction(H_store['periodic_spike_jittered'], np.array(bin_heights['naive_periodic']), fraction=match_frac)[0]*np.array(bin_heights['naive_periodic']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0
        results_short['naive_CC_bursty_spike_jitter'] = 1 if multiple_test_poisson(H_store['bursty_spike_jittered'], scale_for_fraction(H_store['bursty_spike_jittered'], np.array(bin_heights['naive_bursty']), fraction=match_frac)[0]*np.array(bin_heights['naive_bursty']), alpha=sig_alpha, method="bonferroni", sided="two", surrogate_max=True, n_surrogates=1000)['surrogate_global_significant'] else 0


        # ## calculate granger causality on original (non jittered data) and save results_short
        bin_width_GC = 3
        results_short['GC_no_spike'] = granger_min_pvalue_from_events(t2_no_plus_spike, t1_no, bin_width_GC,
                                        maxlag=5, alpha=0.05,
                                        align_method='intersection',
                                        verbose=False)
        results_short['GC_periodic'] = granger_min_pvalue_from_events(t2_periodic_plus_spike, t1_periodic, bin_width_GC,
                                        maxlag=5, alpha=0.05,
                                        align_method='intersection',
                                        verbose=False)
        results_short['GC_bursty'] = granger_min_pvalue_from_events(t2_bursty_plus_spike, t1_bursty, bin_width_GC,
                                        maxlag=5, alpha=0.05,
                                        align_method='intersection',
                                        verbose=False)
        
        results_short['alpha'] = alpha

        for k, v in results_short.items():
            results[k].append(v)

# save results
with open(f'calc_TOD_simulation_results_alpha_varying_jitter_py_{jitter_window}_int{int_l}_{int_u}_BR{BR}_c1_{c1}_c2_{c2}_n{n_iter}.pkl', 'wb') as f:
    pickle.dump(results, f)


results_final = {}
for k, vlist in results.items():
    try:
        results_final[k] = np.array(vlist)
    except Exception:
        # fallback: keep as list if conversion fails
        results_final[k] = vlist

# save results_final
fname = f'calc_TOD_final_simulation_results_alpha_varying_jitter_py_{jitter_window}_int{int_l}_{int_u}_BR{BR}_c1_{c1}_c2_{c2}_n{n_iter}.pkl'
with open(fname, 'wb') as f:
    pickle.dump(results_final, f)