
"""
Preamble for most code and jupyter notebooks
@author: bridgetsmart
@notebook start date: 19 Apr 2023
"""

import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

import math, string, re, pickle, json, time, os, sys, datetime, itertools

from tqdm.notebook import tqdm
from collections.abc import Iterable

# need a function to flatten irregular list of lists
def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def possible_time_del(t1,t2,win,T):
    '''
    Here window truncates the distribution so you don't see all the edges
    one sided - source to target

    possible speed up using numba
    '''
    deltas_ = []
    for x in t1:
        if ((win)<=x<=(T-win)):  # exclude boundaries of size w - only want the middle T-w 'win' of the data
            for y in t2:
                if abs(x-y) <= win:
                    deltas_.append(-x+y)
    return deltas_

def plot_cross_correlogram(source,target,win,T,delta,ylim=None, ax=None, return_H = False, lab=False, title = None):
    '''
    delta = width of hist bin

    '''
    if type(source) == list:
        source = np.array(source)
    if type(target)==list:
        target = np.array(target)

    bins= np.arange(-win,win+delta,delta)
    to_zero = np.where((bins == -delta))[0][0], np.where(bins==0)[0][0]

    if not ax:
        fig, ax = plt.subplots()
    deltas = possible_time_del(source,target,win,T)
    # print('ob deltas')
    H, bins = np.histogram(deltas, bins= np.arange(-win,win+delta,delta))
    Hr = H.copy()
    for ind in to_zero:
        H[ind] = 0
    if not lab:
        ax.bar(bins[:-1], H, width=np.diff(bins), align='center')
    else:
        ax.bar(bins[:-1], H, width=np.diff(bins), align='center', label='observed')

    if not title:
        ax.set_title(rf'Cross-correlogram, $\delta$')
    else:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ax.set_ylim(0.8*np.min(H),1.1*np.max(H))



    # plt.show()
    if return_H:
        return Hr, ax
    else:
        return ax

def deltas_cross_correlogram(source,target,win,T):
    '''
    delta = width of hist bin

    '''
    return possible_time_del(source,target,win,T)

def left_bin_edges(win, delta):
    return np.arange(-win,win+delta,delta)[:-1]


def plot_cc_from_deltas(deltas, win, delta,ylim=None, ax=None, alpha = 1, n_iter = 1, return_H=False, H_no_plot = False, title=None):
    '''
    deltas should be a list of results from different trials
    [trial 1 deltas, trial 2 deltas, ...]
    '''
    if not ax:
        fig, ax = plt.subplots()

    H, bins = np.histogram(deltas, bins= np.arange(-win,win+delta,delta))

    if H_no_plot:
        return H
    ax.bar(bins[:-1], H, width=np.diff(bins), align='edge', alpha=alpha)
    if not title:
        ax.set_title(rf'Cross-correlogram, $\delta$')
    else:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ax.set_ylim(0.8*np.min(H),1.1*np.max(H))

    if not return_H:
        return ax
    else:
        return ax, H
    


## EXAMPLE FUNCTIONS
def run_single_test(t1, t2, df, win, delta, T, tag_A = "A", tag_B = "B", factor_red = 10):

    fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(14,3.3))
    # increase spacing between subplots
    fig.subplots_adjust(wspace=0.4)

    res = {}

    if type(t1) == list:
        t1 = np.array(t1)
    if type(t2)==list:
        t2 = np.array(t2)


    t1 = t1[t1>=win]
    t1 = t1[t1<=T-win]

    bins= np.arange(-win,win+delta,delta)
    to_zero = np.where((bins == -delta))[0][0], np.where(bins==0)[0][0]


    deltas = []

    for t in tqdm(np.array_split(t1, factor_red)): # auspol occurs 3.6 times more than the second most frequent - this is a safe value
        deltas.append(possible_time_del(t, t2, win, T))
    
    deltas = np.array(list(flatten(deltas)))

    ylim=None
    return_H = True
    lab = True

    title = None


    H, bins = np.histogram(deltas, bins= np.arange(-win,win+delta,delta))

    Hr = H.copy()

    for ind in to_zero:
        H[ind] = 0
    
    axs[0].bar(bins[:-1], H, width=np.diff(bins), align='center', label='observed')
    axs[2].bar(bins[:-1], H, width=np.diff(bins), align='center')

    # fit poisson ih
    lambda_1 = len(t1) / (T-2*win)
    lambda_2 = len(t2) / (T-2*win)

    # significance lines

    bins = np.arange(-win,win+delta,delta)
    b_heights = [expected_bin_height(x, lambda x : lambda_1,  lambda x : lambda_2, delta, win, T) for x in left_bin_edges(win, delta)]
    res['no_tod_lambdas'] = list(b_heights)
    axs[0].bar(bins[:-1], b_heights, width=np.diff(bins), align='center', label = 'theoretical', color='red', alpha=0.5)
    b_heights = np.array(b_heights)
    u = b_heights+3*np.sqrt(b_heights)
    l = b_heights-3*np.sqrt(b_heights)
    sns.lineplot(x=bins[:-1],y=b_heights+3*np.sqrt(b_heights), ax=axs[0],linestyle='--',color='red')
    sns.lineplot(x=bins[:-1], y=b_heights-3*np.sqrt(b_heights), ax=axs[0],linestyle='--',color='red', label = r'$\sigma$')

    axs[0].set_title(f'Cross-correlogram')#\nSignificance value: {np.sum([get_dist(lt,x,ut) for lt,x,ut in zip(l,H,u)]) / np.sum(H)}')

    # set y limits
    axs[0].set_ylim([np.max([0.95*np.min(b_heights-3*np.sqrt(b_heights)),0,0.95*np.min(H)]),1.05*np.max([np.max(H),np.max(b_heights+3*np.sqrt(b_heights))])])
    axs[0].set_xlabel('Inter-event times')
    
    ticks = np.arange(-win,win+delta, 60*30) # Now we know have many ticks you need
    labels = ["%.0f" % (x/(60)) for x in ticks] # Generate tick labels
    # xticks = ticks # and the actual positions of the ticks
    axs[0].set_xticks(ticks, labels=labels)
    axs[0].set_ylabel('Occurances')
    # ########################################

    # # Plot 2
    sns.histplot(data = df, x = 'hour', bins=24, hue='tag', alpha=0.2, ax=axs[1], palette=['red','#1f77b4'])
    # source
    x=np.arange(0,24,1)
    y1=df[df.tag==tag_A].groupby('hour').count()['tag'].reindex(index=x, fill_value=0).values
    y_min = np.min(y1)

    def func(x,a,c):
        return a*(np.sin(x/(24/(2*np.pi))+(18/(np.pi)))+c) # period of one day


    def func_with_pen(x,a,c):
        return func(x, a, c)

    fit_p1,_ = curve_fit(func_with_pen, x, y1)
    sns.lineplot(x=x, y=func_with_pen(x, *fit_p1), color='red', ax=axs[1], linewidth=3)
    mse1 = np.mean((y1-func_with_pen(x, *fit_p1))**2)
    # target
    y2=df[df.tag==tag_B].groupby('hour').count()['tag'].reindex(index=x, fill_value=0).values
    y_min2 = np.min(y1)

    def func(x,a,c):
        return a*(np.sin(x/(24/(2*np.pi))+(14/(np.pi)))+c) # period of one day


    def func_with_pen(x,a,c):
        return func(x, a, c)
    fit_p2,_ = curve_fit(func_with_pen, x, y2)
    sns.lineplot(x=x, y=func_with_pen(x, *fit_p2), color='#1f77b4', ax=axs[1], linewidth=3)
    mse2 = np.mean((y2-func_with_pen(x, *fit_p2))**2)
    axs[1].set_title(f"TOD model fit")#\nMSE for #{tag_A} : {round(mse1,3)},\t\t\t\t\t MSE for #{tag_B} : {round(mse2,3)}")

    # ##################################################
    # # plot 3 - cross-correlogram, no TOD effect

    def lambda_s(x):
        # divide by number of days in dataset
        return fit_p2[0]*(np.sin(x/(24/(2*np.pi))+(18/(np.pi)))+fit_p2[1]) /(T//24)

    def lambda_t(x):
        # divide by number of days in dataset
        return fit_p2[0]*(np.sin(x/(24/(2*np.pi))+(14/(np.pi)))+fit_p2[1])  /(T//24)

    # H, _ = plot_cross_correlogram(np.array(t1),np.array(t2),win,T,delta,ylim=None, ax=axs[2], return_H = True, lab=True)
    # res['tod_h'] = list(H)
    # significance lines

    bins = np.arange(-win,win+delta,delta)
    b_heights_tod = [expected_bin_height(x, lambda_s,  lambda_t, delta, win, 14*T)/14 for x in left_bin_edges(win, delta)]
    res['tod_lambdas'] = list(b_heights_tod)

    axs[2].bar(bins[:-1], b_heights_tod, width=np.diff(bins), align='center', label = 'theoretical', color='red', alpha=0.5)
    b_heights_tod = np.array(b_heights_tod)
    u2 = b_heights_tod+3*np.sqrt(b_heights_tod)
    l2 = b_heights_tod-3*np.sqrt(b_heights_tod)
    sns.lineplot(x=bins[:-1],y=b_heights_tod+3*np.sqrt(b_heights_tod), ax=axs[2],linestyle='--',color='red')
    sns.lineplot(x=bins[:-1], y=b_heights_tod-3*np.sqrt(b_heights_tod), ax=axs[2],linestyle='--',color='red')

    axs[2].set_title(f'Cross-correlogram including TOD effect')#\nSignificance value: {np.sum([get_dist(lt,x,ut) for lt,x,ut in zip(l,H,u)]) / np.sum(H)}')

    # set y limits
    axs[2].set_ylim([np.max([0.95*np.min(b_heights_tod-3*np.sqrt(b_heights_tod)),0,0.95*np.min(H)]),1.05*np.max([np.max(H),np.max(b_heights_tod+3*np.sqrt(b_heights_tod))])])
    axs[2].set_xlabel('Inter-event times')
    axs[2].set_ylabel('Occurances')

    # # set all the font sizes to large
    # for ax in axs:
    #     ax.tick_params(axis='both', which='major', labelsize=10)
    #     ax.tick_params(axis='both', which='minor', labelsize=10)
    #     ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    #     ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    #     ax.set_title(ax.get_title(), fontsize=15)

    axs[0].legend(loc='upper center', bbox_to_anchor=(1.7, -0.17), shadow=True, ncol=3)

    axs[2].legend().set_visible(False)

    plt.savefig(f'{tag_A}_{tag_B}2.pdf', bbox_inches='tight')
    # save res
    # res to dataframe
    # save res
    # pd.DataFrame(res).to_csv(f'{tag_A}_{tag_B}_bin_heights.csv', index=False)
    return H, bins, u, l, u2, l2


from scipy.optimize import curve_fit

import scipy.integrate as integrate
def expected_target_events_hj_delta(si, hj, delta, lambda_T):
    l= si + hj
    return integrate.quad(lambda_T, l, l+delta)[0] # integral from si+hj to si+hj+delta of lambda_T

def expected_bin_height(hj, lambda_S, lambda_T, delta, w, T):
    term = lambda x : lambda_S(x) * expected_target_events_hj_delta(x, hj, delta, lambda_T)
    return integrate.quad(term, w, T-w)[0] # integral from si+hj to si+hj+delta of lambda_S * E[events from T]

def sd_bin_height(hj, lambda_S, lambda_T, delta, w, T):
    term = lambda x : lambda_S(x) * expected_target_events_hj_delta(x, hj, delta, lambda_T)
    return integrate.quad(term, w, T-w)[0] # integral from si+hj to si+hj+delta of lambda_S * E[events from T]
def exp_bin_heighed_jittered(x, lambda_s, lambda_t_jittered, Delta, w, T,jitter_intervals):
    start = w
    end = T-w

    s = 0

    # get index of first jitter interval bigger than start
    i_start = np.where(jitter_intervals >= start)[0][0]    
    i_end = np.where(jitter_intervals >= (T-w))[0][0] -1

    for i in range(i_start, i_end):
        s += expected_bin_height(x, lambda_s, lambda_t_jittered, Delta, jitter_intervals[i] + 1e-6, jitter_intervals[i+1]- 1e-6) 

    if (T-w) > jitter_intervals[i_end]:
        s += expected_bin_height(x, lambda_s, lambda_t_jittered, Delta, jitter_intervals[i_end]+ 1e-6, T-w)

    return s


from scipy.stats import chisquare
def smooth_example_plot(t1,t2, win, delta, T, interval_size, factor_red = 8):
    if type(t1) == list:
        t1 = np.array(t1)
    if type(t2)==list:
        t2 = np.array(t2)


    t1 = t1[t1>=win]
    t1 = t1[t1<=T-win]

    bins= np.arange(-win,win+delta,delta)
    to_zero = np.where((bins == -delta))[0][0], np.where(bins==0)[0][0]


    deltas = []

    for t in np.array_split(t1, factor_red): # auspol occurs 3.6 times more than the second most frequent - this is a safe value
        deltas.append(possible_time_del(t, t2, win, T))
    
    deltas = np.array(list(flatten(deltas)))

    ylim=None
    return_H = True
    lab = True

    title = None

    res = {}
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(12,4.3))

    H, bins = np.histogram(deltas, bins= np.arange(-win,win+delta,delta))

    Hr = H.copy()

    for ind in to_zero:
        H[ind] = 0
    
    if not lab:
        axs[0].bar(bins[:-1], H, width=np.diff(bins), align='center')
    else:
        axs[0].bar(bins[:-1], H, width=np.diff(bins), align='center', label='observed')

    # if not title:
    #     axs[0].set_title(rf'Cross-correlogram, $\delta$')
    # else:
    #     axs[0].set_title(title)
    if ylim:
        axs[0].set_ylim(ylim[0],ylim[1])
    else:
        axs[0].set_ylim(0.8*np.min(H),1.1*np.max(H))

    res['no_jitter_h'] = list(H)
    # fit poisson homogeneous
    lambda_1 = len(t1) / (np.max(t1) - np.min(t1))
    lambda_2 = len(t2) / (np.max(t2) - np.min(t2))

    # significance lines


    bins = np.arange(-win,win+delta,delta)
    b_heights = [expected_bin_height(x, lambda x : lambda_1,  lambda x : lambda_2, delta, win, T) for x in left_bin_edges(win, delta)]
    res['no_tod_lambdas'] = list(b_heights)
    axs[0].bar(bins[:-1], b_heights, width=np.diff(bins), align='center', label = 'theoretical', color='red', alpha=0.5)
    b_heights = np.array(b_heights)
    u = b_heights+3*np.sqrt(b_heights)
    l = b_heights-3*np.sqrt(b_heights)
    sns.lineplot(x=bins[:-1],y=b_heights+3*np.sqrt(b_heights),linestyle='--',color='red', ax=axs[0], label = r'$\sigma$')
    sns.lineplot(x=bins[:-1], y=b_heights-3*np.sqrt(b_heights),linestyle='--',color='red', ax=axs[0])

    sig1v = np.array([get_dist(lt,x,ut) for lt,x,ut in zip(l,H,u)])
    for ind in to_zero:
        sig1v[ind] = 0
    sig1 = np.sum(sig1v) / np.sum(H)
    # p value for poisson null model

    obs = np.array(H)
    obs = np.delete(obs,to_zero)
    exp = np.delete(b_heights,to_zero)
    sig2 = chisquare(obs*np.sum(exp)/np.sum(obs),exp)
    df = len(H)-1 - len(to_zero)

    # set y limits
    y_limits_og = [np.min([0.95*np.min(b_heights-3*np.sqrt(b_heights)),0,0.95*np.min(H)]),1.05*np.max([np.max(H),np.max(b_heights+3*np.sqrt(b_heights))])]

    
    # axs[0].set_xlabel('Offset (hours)')
    axs[0].set_ylabel('Occurances')

    axs[0].set_xlabel('Offset (Minutes)')
    ticks = np.arange(-win,win+delta, 60*30) # Now we know have many ticks you need
    labels = ["%.0f" % (x/(60)) for x in ticks] # Generate tick labels
    # xticks = ticks # and the actual positions of the ticks
    axs[0].set_xticks(ticks, labels=labels, fontsize='small')

    axs[0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.17), shadow=True, ncol=2)

    axs[0].set_title(f'Constant null model', fontsize='medium',ha='left', loc='left')

    ##################################################
    # second jittered plot

    s_times = np.array(t1)
    t_times = np.array(t2)
    
    def lambda_func(x, i__, r__):
        if x < win:
            return 0
        if x > (T-win):
            return 0
        else: 
            return r__[i__>=x][0]
            

    i_s, r_s = get_rates(s_times, interval_size, win,np.max([np.max(s_times),T-win])+2*interval_size)
    i_t, r_t = get_rates(t_times, interval_size, win, np.max([np.max(t_times),T-win])+2*interval_size)

    lambda_s = lambda x : lambda_func(x, i_s, r_s)
    lambda_t = lambda x : lambda_func(x, i_t, r_t)

    # plot observed
    if not lab:
        axs[1].bar(bins[:-1], H, width=np.diff(bins), align='center')
    else:
        axs[1].bar(bins[:-1], H, width=np.diff(bins), align='center', label='observed')

    # if not title:
    #     axs[1].set_title(rf'Cross-correlogram, $\delta$')
    # else:
    #     axs[1].set_title(title)
    if ylim:
        axs[1].set_ylim(ylim[0],ylim[1])
    else:
        axs[1].set_ylim(0.8*np.min(H),1.1*np.max(H))



    res['jitter_h'] = list(H)


    bins = np.arange(-win,win+delta,delta)
    b_heights_smooth = [expected_bin_height(x, lambda_s,  lambda_t, delta, win, T) for x in left_bin_edges(win, delta)]
    res['tod_lambdas'] = list(b_heights_smooth)
    axs[1].bar(bins[:-1], b_heights_smooth, width=np.diff(bins), align='center', label = 'theoretical', color='red', alpha=0.5)
    # b_heights_smooth = b_heights_smooth[1:] + [b_heights_smooth[-1]]
    b_heights_smooth = np.array(b_heights_smooth)
    u = b_heights_smooth+3*np.sqrt(b_heights_smooth)
    l = b_heights_smooth-3*np.sqrt(b_heights_smooth)
    sns.lineplot(x=bins[:-1],y=b_heights_smooth+3*np.sqrt(b_heights_smooth), ax=axs[1],linestyle='--',color='red')
    sns.lineplot(x=bins[:-1], y=b_heights_smooth-3*np.sqrt(b_heights_smooth), ax=axs[1],linestyle='--',color='red')

    sig3v = np.array([get_dist(lt,x,ut) for lt,x,ut in zip(l,H,u)]) / np.sum(H)
    for ind in to_zero:
        sig3v[ind]=0
        
    sig3 = np.sum(sig3v**2) / np.sum(H)
    midpoints = (bins[:-1] + delta/2)
    sig4 = midpoints[np.argmax(sig3v)]
    sig5 = np.average(midpoints, weights = sig3v)
    

    axs[1].set_title(f'Inhomogeneous null model', fontsize='medium', ha='left', loc = 'left')
    # set y limits
    axs[1].set_xlabel('Offset (Minutes)')
    ticks = np.arange(-win,win+delta, 60*30) # Now we know have many ticks you need
    labels = ["%.0f" % (x/(60)) for x in ticks] # Generate tick labels
    # xticks = ticks # and the actual positions of the ticks
    axs[1].set_xticks(ticks, labels=labels, fontsize='small')

    axs[1].set_ylabel('Occurances')

    # turn off legend on ax1
    axs[1].legend().set_visible(False)


    #### y limits
    y_limits_new = [np.min([0.95*np.min(b_heights_smooth-3*np.sqrt(b_heights_smooth)),0,0.95*np.min(H)]),1.05*np.max([np.max(H),np.max(b_heights_smooth+3*np.sqrt(b_heights_smooth))])]

    y_limits_ = [np.min([y_limits_og[0], y_limits_new[0]]), np.max([y_limits_og[1], y_limits_new[1]])]
    
    axs[0].set_ylim(y_limits_)

    axs[1].set_ylim(y_limits_)
    ##################################################
    # plt.suptitle(fr'#{tag_A} $\rightarrow$ #{tag_B}', y=1.08, x=0.48)

    axs[0].legend(loc='upper center', bbox_to_anchor=(1, -0.17), shadow=True, ncol=3)
    
    # save figures
    plt.savefig(f'A_B_smooth1.pdf', bbox_inches='tight')

    return True #float(np.sum(H)),sig1, sig2[0],sig2[1], sig3, sig4, sig5, df, list([int(x) for x in Hr])

# helper functions for smooth example plot
def get_dist(l,x,u):
    if x<l:
        return l-x
    if x>u:
        return x-u
    else:
        return 0


def get_rates(t, interval_size, t_min, t_max):
    intervals = np.arange(t_min, t_max, interval_size)
    rates = []
    for i in range(len(intervals)-1):
        rates.append(len(t[(t>=intervals[i]) & (t<intervals[i+1])]) / interval_size)
    return intervals[:-1], np.array(rates)

    
def lambda_func(x, i__, r__):
    if x < win:
        return 0
    if x > (T-win):
        return 0
    else: 
        return r__[i__>=x][0]