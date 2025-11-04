
""" 
Preamble for most code and jupyter notebooks 
@author: bridgetsmart 

@notebook date: 16th Jul 2023 
"""  
import numpy as np, pandas as pd  
import matplotlib.pyplot as plt, seaborn as sns 
import matplotlib as mpl  
import math, string, re, pickle, json, time, os, sys, datetime, itertools  
from tqdm.notebook import tqdm

plt.rcParams["font.family"] = ['Apple SD Gothic Neo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # to make - render on xlabels


from cross_correlogram import *
from simulating_hawkes import *
from collections.abc import Iterable

# circadian / time of day effect

from scipy.optimize import curve_fit
import scipy.integrate as integrate

from collections import Counter


# Plot colours
red_c = (0.79,0.12,0.42)
blue_exp = "royalblue"#(0,1,1)
tod_1 = "tab:cyan"#(0,0.44,1)
tod_2 = "orange"#(0.2,0.8,0.8)#(0.33,0.87,0.2)
a_c = 0.65

# general functions

def get_new_time(expected_time):
    # generate time between poisson events given the rate.
    # expected time = rate (expected time between events)

    return np.random.exponential(expected_time)

def flatten(xs):
    # function to flatten a list of lists or other iterables
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


# calculating parts of cross-correlogram
def expected_target_events_hj_delta(si, hj, delta, lambda_T):
    l= si + hj
    return integrate.quad(lambda_T, l, l+delta)[0] # integral from si+hj to si+hj+delta of lambda_T

def expected_bin_height(hj, lambda_S, lambda_T, delta, w, T):
    term = lambda x : lambda_S(x) * expected_target_events_hj_delta(x, hj, delta, lambda_T)
    return integrate.quad(term, w, T-w)[0] # integral from si+hj to si+hj+delta of lambda_S * E[events from T]

def sd_bin_height(hj, lambda_S, lambda_T, delta, w, T):
    term = lambda x : lambda_S(x) * expected_target_events_hj_delta(x, hj, delta, lambda_T)
    return integrate.quad(term, w, T-w)[0] # integral from si+hj to si+hj+delta of lambda_S * E[events from T]

# get absolute distance
def get_dist(l,x,u):
    if x<l:
        return l-x
    if x>u:
        return x-u
    else:
        return 0
    
def bursty(lambda_t, BR, int_l, int_u, T):
    """
    Simulates a Poisson process with bursting activity (no refractory period).
    
    Parameters:
    lambda_t: function -- time-dependent rate function for the Poisson process
    BR: float -- burst fraction (probability of a burst following a spike)
    int_l: float -- lower bound of burst inter-spike intervals (ms)
    int_u: float -- upper bound of burst inter-spike intervals (ms)
    T: float -- total simulation time (seconds)
    
    Returns:
    times: list of event times (in seconds)
    """
    
    t_current = 0  # Current time (seconds)
    times = []  # List to store event times

    while t_current < T:
        # Generate Poisson event
        t_n = gen_poisson_homogeneous_single(lambda_t(t_current))
        t_current += t_n

        if t_current >= T:
            break
        
        # Add Poisson event
        times.append(t_current)

        # Handle burst events
        times, t_current = add_new_time(t_current, times, BR, int_l, int_u, T)

    return times


def add_new_time(t_current, times, BR, int_l, int_u, T):

    u = np.random.rand()

    while u < BR:
        # generate a burst event
        t_burst = t_current + np.random.uniform(int_l, int_u)

        if t_burst >= T:
            return times
        
        times.append(t_burst)
        t_current = times[-1]
        u = np.random.rand()
    
        
    return times, t_current


def inactive_bursty(lambda_t, BR, int_l, int_u, T, c_t):
    t_current = 0
    times = []
    while t_current < T:

        t_n = gen_poisson_homogeneous_single(lambda_t(t_current))
        t_current += t_n

        times = add_new_time_inactive(t_current, times, lambda_t, BR, int_l, int_u, T, c_t)

        t_current = times[-1]
    
    return times

def add_new_time_inactive(t_current, times, lambda_t, BR, int_l, int_u, T, c_t):

    times.append(t_current)

    u = np.random.rand()
    active_bursty, _ = get_current_state(c_t, times[-1])

    while (u < BR) and (active_bursty==1):
        times.append(t_current + np.random.uniform(int_l, int_u))
        t_current = times[-1]

        active_bursty, _ = get_current_state(c_t, times[-1])
        u = np.random.rand()

        if t_current >= T:
            return times
        
    return times

def sim_ct(rates, t_finish, t_current, current_state = 0):
    # simulating c(t) for a single process
    # c(t) is the underlying markov process which 
    # describes if a process is currently coordinating
    # or not.

    # rates = vector giving 
    #   [expected time to move into coordinated state,
    #    expected time to move into a non coordinated state]

    # t_finish = time to simulate events until
    # t_current = time to simulate events from

    # current_state = 0 # non coord is default

    transition_times = []
    t_dash = get_new_time(rates[current_state])
    t_current += t_dash
    current_state = np.mod(current_state+1,2)
    # generate exponential random variable with rate 1/to_coord
    while t_current<t_finish:
        transition_times.append(t_current)
        t_dash = get_new_time(rates[current_state])

        t_current += t_dash
        current_state = np.mod(current_state+1,2)


    return [k for k,v in Counter(transition_times).items() if v==1] # dict preserves order

def get_current_state(c_t, current_time):
    #### Modified ####
    # returns 0 if not coordinating
    # returns 1 otherwise
    ind_test = bisect.bisect_left(c_t, current_time)
    if ind_test == len(c_t):
        return np.mod(ind_test,2), False
    else:
        return np.mod(ind_test,2), c_t[ind_test]

# get the average rate for each interval and return as a np array
def get_rates(t, interval_size, t_min, t_max):
    intervals = np.arange(t_min, t_max, interval_size)
    rates = []
    for i in range(len(intervals)-1):
        rates.append(len(t[(t>=intervals[i]) & (t<intervals[i+1])]) / interval_size)
    return intervals[:-1], np.array(rates)

def jitter_times(t, interval_size, t_min, t_max):
    intervals = np.arange(t_min, t_max, interval_size)
    jittered_times = []
    for i in range(len(intervals)-1):
        jittered_times.append(intervals[i] + np.random.rand(len(t[(t>=intervals[i]) & (t<intervals[i+1])]))*interval_size)

    jittered_times.append(intervals[-1] + np.random.rand(len(t[(t>=intervals[-1])]))*interval_size)

    return np.array(list(flatten(jittered_times)))
    
def get_plots_periodic(t1, t2, tj_target, tj_source, win, delta, interval_sizes, jitter_win_sizes, T, factor_red = 10):
    fig, axs = plt.subplots(nrows=3, ncols=3,figsize=(12.5,12))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axs = axs.ravel()

    j_row_offset = 6

    t1 = t1[t1>=win]
    t1 = t1[t1<=T-win]

    bins= np.arange(-win,win+delta,delta)
    to_zero = []#np.where((bins == -delta))[0][0], np.where(bins==0)[0][0]

    # Now to calculate all bins for cross-correlograms
    deltas = []
    deltas_quarter = []
    deltas_half = []
    deltas_one = []

    for t in tqdm(np.array_split(t1, factor_red)): # auspol occurs 3.6 times more than the second most frequent - this is a safe value
        deltas.append(possible_time_del(t, t2, win, T))

    for t in tqdm(np.array_split(tj_source[0], factor_red)):
        deltas_quarter.append(possible_time_del(t, tj_target[0], win, T))
    
    for t in tqdm(np.array_split(tj_source[1], factor_red)):
        deltas_half.append(possible_time_del(t, tj_target[1], win, T))

    for t in tqdm(np.array_split(tj_source[2], factor_red)):
        deltas_one.append(possible_time_del(t, tj_target[2], win, T))
    
    deltas = np.array(list(flatten(deltas)))
    jitter_deltas = [deltas_quarter, deltas_half, deltas_one]


    # start by calculating the unmodified cross-correlogram
    H, bins = np.histogram(deltas, bins= np.arange(-win,win+delta,delta))

    # for ind in to_zero:
    #     H[ind] = 0

    # plot on plots 0, 1, 2
    axs[0].bar(bins[:-1], H, width=np.diff(bins), align='center', label='observed', color = blue_exp)
    axs[2].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)
    axs[3].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)
    axs[4].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)
    axs[5].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)


    min_H_jitter = 1e6
    max_H_jitter = 0

    for i in range(3):
        deltas_ = np.array(list(flatten(jitter_deltas[i])))
        H_, bins = np.histogram(deltas_, bins= np.arange(-win,win+delta,delta))

        axs[j_row_offset+i].bar(bins[:-1], H_, width=np.diff(bins), align='center', label='observed', color = blue_exp)

        min_H_jitter = np.min([min_H_jitter,np.min(H_)])
        max_H_jitter = np.max([max_H_jitter,np.max(H_)])
    

    ##### FIRST ROW - NO JITTER SIGNIFICANCE LINES

    ## Homogenous Poisson null model
    lambda_phs = len(t1) / (T-2*win)
    lambda_pht = len(t2) / (T-2*win)

    ## plot significance lines on axs 0
    b_heights_ph = [expected_bin_height(x, lambda x : lambda_phs,  lambda x : lambda_pht, delta, win, T) for x in left_bin_edges(win, delta)]
    b_heights_ph = np.array(b_heights_ph)
    axs[0].bar(bins[:-1], b_heights_ph, width=np.diff(bins), align='center', label = 'theoretical', color=red_c, alpha=0.8)
    sns.lineplot(x=bins[:-1],y=b_heights_ph+3*np.sqrt(b_heights_ph), ax=axs[0],linestyle='--',color=red_c)
    sns.lineplot(x=bins[:-1], y=b_heights_ph-3*np.sqrt(b_heights_ph), ax=axs[0],linestyle='--',color=red_c, label = r'$\sigma$')


    ## plot functional TOD model on axs 1
    x=np.arange(0,25,1)

    ys = np.mod(t1//1,24)
    yt = np.mod(t2//1,24)

    # plot hours
    vs, _= np.histogram(ys, bins=x)
    vt, _= np.histogram(yt, bins=x)

    vs = vs / (T//24)
    vt = vt / (T//24)

    x = np.arange(0,24,1)

    axs[1].bar(x,vs,width=1,align='edge',color=tod_1, alpha=0.35, label = 'source')
    axs[1].bar(x,vt,width=1,align='edge',color=tod_2, alpha=0.35, label = 'target')

    # source
    def func(x,a,c):
        return a*(np.sin(x/(24/(2*np.pi))+(18/(np.pi)))+c) # period of one day


    def func_with_pen(x,a,c):
        return func(x, a, c)


    fit_p1,_ = curve_fit(func_with_pen, x,vs)
    sns.lineplot(x=x, y=func_with_pen(x, *fit_p1), color=tod_1, ax=axs[1], linewidth=3)

    # target

    def func(x,a,c):
        return a*(np.sin(x/(24/(2*np.pi))+(14/(np.pi)))+c) # period of one day


    def func_with_pen(x,a,c):
        return func(x, a, c)

    fit_p2,_ = curve_fit(func_with_pen, x, vt)
    sns.lineplot(x=x, y=func_with_pen(x, *fit_p2), color=tod_2, ax=axs[1], linewidth=3)


    def lambda_tod_s(x):
        # divide by number of days in dataset
        return (fit_p2[0]*(np.sin(x/(24/(2*np.pi))+(18/(np.pi)))+fit_p2[1])) 

    def lambda_tod_t(x):
        # divide by number of days in dataset
        return (fit_p2[0]*(np.sin(x/(24/(2*np.pi))+(14/(np.pi)))+fit_p2[1]) ) 

    ## Plot cross-correlogram with TOD effect on axs 2

    bins = np.arange(-win,win+delta,delta)
    b_heights_tod = [expected_bin_height(x, lambda_tod_s,  lambda_tod_t, delta, win, T) for x in left_bin_edges(win, delta)]

    axs[2].bar(bins[:-1], b_heights_tod, width=np.diff(bins), align='center', label = 'theoretical', color=red_c, alpha=a_c)
    b_heights_tod = np.array(b_heights_tod)
    sns.lineplot(x=bins[:-1],y=b_heights_tod+3*np.sqrt(b_heights_tod), ax=axs[2],linestyle='--',color=red_c)
    sns.lineplot(x=bins[:-1], y=b_heights_tod-3*np.sqrt(b_heights_tod), ax=axs[2],linestyle='--',color=red_c)
    

    ## Plot smoothed cross-correlogram on axs 3
    s_times = np.array(t1)
    t_times = np.array(t2)
    b_heights_smooth = []
    max_r2 = []
    min_r2 = []
    
    for i in range(3):
        interval_size = interval_sizes[i]
        i_s, r_s = get_rates(s_times, interval_size, win,np.max([np.max(s_times),T-win])+2*interval_size)
        i_t, r_t = get_rates(t_times, interval_size, win, np.max([np.max(t_times),T-win])+2*interval_size)

        def lambda_func(x, i__, r__):
            if x < win:
                return 0
            if x > (T-win):
                return 0
            else: 
                return r__[i__>=x][0]

        lambda_inhs = lambda x : lambda_func(x, i_s, r_s)
        lambda_inht = lambda x : lambda_func(x, i_t, r_t)

        # plot significance lines
        b_heights_smooth.append(np.array([expected_bin_height(x, lambda_inhs,  lambda_inht, delta, win, T) for x in left_bin_edges(win, delta)]))

        axs[3+i].bar(bins[:-1], b_heights_smooth[i], width=np.diff(bins), align='center', color=red_c, alpha=a_c)
        sns.lineplot(x=bins[:-1],y=b_heights_smooth[i]+3*np.sqrt(b_heights_smooth[i]), ax=axs[3+i],linestyle='--',color=red_c)
        sns.lineplot(x=bins[:-1], y=b_heights_smooth[i]-3*np.sqrt(b_heights_smooth[i]), ax=axs[3+i],linestyle='--',color=red_c)

        max_r2.append(np.max(b_heights_smooth[i]+3*np.sqrt(b_heights_smooth[i])))
        min_r2.append(np.min(b_heights_smooth[i]-3*np.sqrt(b_heights_smooth[i])))



    ###### SECOND ROW - JITTER
    # plot jittered cross-correlograms
    b_heights_jitter = []
    max_j = []
    min_j = []

    for i in range(3):
        lambda_1_con = len(tj_source[i]) / (T-2*win)
        lambda_2_con = len(tj_target[i]) / (T-2*win)

        b_heights_jitter.append(np.array([expected_bin_height(x, lambda x : lambda_1_con,  lambda x : lambda_2_con, delta, win, T) for x in left_bin_edges(win, delta)]))
        
        axs[j_row_offset+i].bar(bins[:-1], b_heights_jitter[i], width=np.diff(bins), align='center', label = 'theoretical', color=red_c, alpha=a_c)
    
        sns.lineplot(x=bins[:-1],y=b_heights_jitter[i]+3*np.sqrt(b_heights_jitter[i]), ax=axs[j_row_offset+i],linestyle='--',color=red_c)
        sns.lineplot(x=bins[:-1], y=b_heights_jitter[i]-3*np.sqrt(b_heights_jitter[i]), ax=axs[j_row_offset+i],linestyle='--',color=red_c, label = r'$\sigma$')

        max_j.append(np.max(b_heights_jitter[i]+3*np.sqrt(b_heights_jitter[i])))
        min_j.append(np.min(b_heights_jitter[i]-3*np.sqrt(b_heights_jitter[i])))


    # y limits

    # Row 1
    min_b_height_r1 = np.max([0,np.min([0.95*np.min(b_heights_tod-3*np.sqrt(b_heights_tod)), 0.95*np.min(b_heights_ph-3*np.sqrt(b_heights_ph)),0.9*np.min(H)])])
    max_b_height_r1 = 1.05*np.max([np.max(H),np.max(b_heights_tod+3*np.sqrt(b_heights_tod)), np.max(b_heights_ph+3*np.sqrt(b_heights_ph))])

    axs[0].set_ylim(min_b_height_r1, max_b_height_r1)
    axs[2].set_ylim(min_b_height_r1, max_b_height_r1)

    # # Row 2
    min_b_height_r2 = 0.95*np.max([np.min(min_r2),0])
    max_b_height_r2 = 1.05*np.max(np.max(max_r2))

    axs[3].set_ylim(min_b_height_r2, max_b_height_r2)
    axs[4].set_ylim(min_b_height_r2, max_b_height_r2)
    axs[5].set_ylim(min_b_height_r2, max_b_height_r2)


    # Row 3
    min_b_height_r3 = 0.95*np.max([np.min([np.min(min_j[1:]), min_H_jitter]),0])
    max_b_height_r3 = 1.05*np.max([np.max(max_j), max_H_jitter])

    axs[6].set_ylim(min_b_height_r3, max_b_height_r3)
    axs[7].set_ylim(min_b_height_r3, max_b_height_r3)
    axs[8].set_ylim(min_b_height_r3, max_b_height_r3)

    ## legend and other formatting

    axs[0].legend(loc='upper center', bbox_to_anchor=(1.8, -2.75), shadow=False, ncol=3)

    for i in np.arange(1,9):
        axs[i].legend().set_visible(False)
        axs[i].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    titles = ['(i) Homogeneous null model',
              '(ii) Circadian model fit',
              '(iii) Functional smooth null model',
              f'(iv) Smooth null model\nInterval size {interval_sizes[0]}',
              f'(v) Smooth null model\nInterval size {interval_sizes[1]}',
              f'(vi) Smooth null model\nInterval size {interval_sizes[2]}',
              f'(vii) Jittered times\nWindow: {jitter_win_sizes[0]}',
              f'(viii) Jittered times\nWindow: {jitter_win_sizes[1]}',
              f'(ix) Jittered times\nWindow: {jitter_win_sizes[2]}']
    
    for i in range(9):
        axs[i].set_title(titles[i])


    plt.savefig('periodic_megaplot.pdf', bbox_inches='tight')

    return True 

##################### BURSTY #################

def get_plots_bursty(t1, t2, tj_target, tj_source, win, delta, interval_sizes, jitter_win_sizes, T, factor_red = 10):

    fig, axs = plt.subplots(nrows=3, ncols=3,figsize=(12.5,12))

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axs = axs.ravel()

    j_row_offset = 6
    
    t1 = t1[t1>=win]
    t1 = t1[t1<=T-win]

    tj_source = [x[x>=win] for x in tj_source]
    tj_source = [x[x<=T-win] for x in tj_source]


    bins= np.arange(-win,win+delta,delta)
    to_zero = []#np.where((bins == -delta))[0][0], np.where(bins==0)[0][0]

    # Now to calculate all bins for cross-correlograms
    deltas = []
    deltas_quarter = []
    deltas_half = []
    deltas_one = []
    deltas_two = []

    for t in tqdm(np.array_split(t1, factor_red)): # auspol occurs 3.6 times more than the second most frequent - this is a safe value
        deltas.append(possible_time_del(t, t2, win, T))

    for t in tqdm(np.array_split(tj_source[0], factor_red)):
        deltas_quarter.append(possible_time_del(t, tj_target[0], win, T))
    
    for t in tqdm(np.array_split(tj_source[1], factor_red)):
        deltas_half.append(possible_time_del(t, tj_target[1], win, T))

    for t in tqdm(np.array_split(tj_source[2], factor_red)):
        deltas_one.append(possible_time_del(t, tj_target[2], win, T))

    for t in tqdm(np.array_split(tj_source[2], factor_red)):
        deltas_two.append(possible_time_del(t, tj_target[2], win, T))

    deltas = np.array(list(flatten(deltas)))
    jitter_deltas = [deltas_quarter, deltas_half, deltas_one, deltas_two]


    # start by calculating the unmodified cross-correlogram
    H, bins = np.histogram(deltas, bins= np.arange(-win,win+delta,delta))

    for ind in to_zero:
        H[ind] = 0

    # plot cross-correlogram for unmodified series
    axs[0].bar(bins[:-1], H, width=np.diff(bins), align='center', label='observed', color = blue_exp)
    axs[1].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)

    axs[3].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)
    axs[4].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)
    axs[5].bar(bins[:-1], H, width=np.diff(bins), align='center', color = blue_exp)


    # min_jitter_H

    min_H_jitter = 1e6
    max_H_jitter = 0
    jitter_plots = [2,6,7,8]

    for i in range(4):
        deltas_ = np.array(list(flatten(jitter_deltas[i])))
        H_, bins = np.histogram(deltas_, bins= np.arange(-win,win+delta,delta))

        axs[jitter_plots[i]].bar(bins[:-1], H_, width=np.diff(bins), align='center', label='observed', color = blue_exp)

        min_H_jitter = np.min([min_H_jitter,np.min(H_)])
        max_H_jitter = np.max([max_H_jitter,np.max(H_)])
    

    ##### FIRST ROW - NO JITTER SIGNIFICANCE LINES

    ## Homogenous Poisson null model
    lambda_phs = len(t1) / (T-2*win)
    lambda_pht = len(t2) / (T-2*win)

    ## plot significance lines on axs 0
    b_heights_ph = [expected_bin_height(x, lambda x : lambda_phs,  lambda x : lambda_pht, delta, win, T) for x in left_bin_edges(win, delta)]
    b_heights_ph = np.array(b_heights_ph)
    axs[0].bar(bins[:-1], b_heights_ph, width=np.diff(bins), align='center', label = 'theoretical', color=red_c, alpha=a_c)
    sns.lineplot(x=bins[:-1],y=b_heights_ph+3*np.sqrt(b_heights_ph), ax=axs[0],linestyle='--',color=red_c)
    sns.lineplot(x=bins[:-1], y=b_heights_ph-3*np.sqrt(b_heights_ph), ax=axs[0],linestyle='--',color=red_c, label = r'$\sigma$')


    ###### SECOND ROW - JITTER
    # plot jittered cross-correlograms

    for i in range(4):

        axs[jitter_plots[i]].bar(bins[:-1], b_heights_ph, width=np.diff(bins), align='center', label = 'theoretical', color=red_c, alpha=a_c)
    
        sns.lineplot(x=bins[:-1],y=b_heights_ph+3*np.sqrt(b_heights_ph), ax=axs[jitter_plots[i]],linestyle='--',color=red_c)
        sns.lineplot(x=bins[:-1], y=b_heights_ph-3*np.sqrt(b_heights_ph), ax=axs[jitter_plots[i]],linestyle='--',color=red_c, label = r'$\sigma$')



    ## Plot smoothed cross-correlogram on axs 1, 3, 4, 5
    ih_plots = [1,3,4,5]
    s_times = np.array(t1)
    t_times = np.array(t2)
    b_heights_smooth = []
    max_r2 = []
    min_r2 = []
    
    for i in range(4):
        interval_size = interval_sizes[i]
        i_s, r_s = get_rates(s_times, interval_size, win,np.max([np.max(s_times),T-win])+2*interval_size)
        i_t, r_t = get_rates(t_times, interval_size, win, np.max([np.max(t_times),T-win])+2*interval_size)

        def lambda_func(x, i__, r__):
            if x < win:
                return 0
            if x > (T-win):
                return 0
            else: 
                return r__[i__>=x][0]

        lambda_inhs = lambda x : lambda_func(x, i_s, r_s)
        lambda_inht = lambda x : lambda_func(x, i_t, r_t)

        # plot significance lines
        b_heights_smooth.append(np.array([expected_bin_height(x, lambda_inhs,  lambda_inht, delta, win, T) for x in left_bin_edges(win, delta)]))

        axs[ih_plots[i]].bar(bins[:-1], b_heights_smooth[i], width=np.diff(bins), align='center', color=red_c, alpha=a_c)
        sns.lineplot(x=bins[:-1],y=b_heights_smooth[i]+3*np.sqrt(b_heights_smooth[i]), ax=axs[ih_plots[i]],linestyle='--',color=red_c)
        sns.lineplot(x=bins[:-1], y=b_heights_smooth[i]-3*np.sqrt(b_heights_smooth[i]), ax=axs[ih_plots[i]],linestyle='--',color=red_c)

        max_r2.append(np.max(b_heights_smooth[i]+3*np.sqrt(b_heights_smooth[i])))
        min_r2.append(np.min(b_heights_smooth[i]-3*np.sqrt(b_heights_smooth[i])))


    ## y limits

    # Row 1
    min_b_height_r1 = np.max([0,np.min([0.95*np.min(b_heights_ph-3*np.sqrt(b_heights_ph)),0.9*np.min(H),min_r2[0]])])
    max_b_height_r1 = 1.05*np.max([np.max(H),np.max([np.max(b_heights_ph+3*np.sqrt(b_heights_ph)),max_r2[0]])])

    axs[0].set_ylim(min_b_height_r1, max_b_height_r1)
    axs[1].set_ylim(min_b_height_r1, max_b_height_r1)
    axs[2].set_ylim(min_b_height_r1, max_b_height_r1)

    # Row 2
    min_b_height_r2 = 0.95*np.max([np.min(min_r2[1:]),0])
    max_b_height_r2 = 1.05*np.max(np.max(max_r2[1:]))

    axs[3].set_ylim(min_b_height_r2, max_b_height_r2)
    axs[4].set_ylim(min_b_height_r2, max_b_height_r2)
    axs[5].set_ylim(min_b_height_r2, max_b_height_r2)


    # Row 3
    min_b_height_r3 = np.max([0,np.min([np.min([0.95*np.min(b_heights_ph-3*np.sqrt(b_heights_ph))]),0.9*np.min(H), min_H_jitter]),0])
    max_b_height_r3 = 1.05*np.max([np.min([0.95*np.max(b_heights_ph-3*np.sqrt(b_heights_ph))]), max_H_jitter])

    axs[6].set_ylim(min_b_height_r3, max_b_height_r3)
    axs[7].set_ylim(min_b_height_r3, max_b_height_r3)
    axs[8].set_ylim(min_b_height_r3, max_b_height_r3)

    ## legend and other formatting

    axs[0].legend(loc='upper center', bbox_to_anchor=(1.8, -2.75), shadow=False, ncol=3)

    for i in np.arange(1,9):
        axs[i].legend().set_visible(False)
        axs[i].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    titles = ['(i) Homogeneous null model',
              f'(ii) Smooth null model\nInterval size {interval_sizes[0]}',
              f'(iii) Jittered times\nWindow: {jitter_win_sizes[0]}',
              f'(iv) Smooth null model\nInterval size {interval_sizes[1]}',
              f'(v) Smooth null model\nInterval size {interval_sizes[2]}',
              f'(vi) Smooth null model\nInterval size {interval_sizes[3]}',
              f'(vii) Jittered times\nWindow: {jitter_win_sizes[1]}',
              f'(viii) Jittered times\nWindow: {jitter_win_sizes[2]}',
              f'(ix) Jittered times\nWindow: {jitter_win_sizes[3]}']
    
    for i in range(9):
        axs[i].set_title(titles[i])

    plt.savefig('bursty_megaplot.pdf', bbox_inches='tight')

    return True 