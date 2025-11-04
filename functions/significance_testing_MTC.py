"""
multiple_test_poisson.py

A self-contained function you can copy-paste that:
 - computes per-bin Poisson p-values for observed counts O given expected counts E
 - applies multiple-comparison correction (Bonferroni or Benjamini-Hochberg FDR)
 - optionally runs a surrogate max-statistic test to give a family-wise global p-value
 - supports one-sided ('greater' or 'less') or two-sided testing

Dependencies:
 - numpy
 - optionally scipy (for exact Poisson tail functions). If scipy is not available,
   the code uses a safe fallback for Poisson cdf computations.
"""

from typing import Sequence, Dict, Any, List, Tuple
import math
import numpy as np

# Try to import scipy for accurate Poisson tail probabilities
try:
    from scipy.stats import poisson
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _poisson_pvals_single(O: int, E: float) -> Dict[str, float]:
    """Return one-sided ('greater','less') and two-sided Poisson p-values for P(X|Pois(E))."""
    if E < 0:
        raise ValueError("Expected count E must be non-negative")
    if E == 0:
        if O == 0:
            return {'greater': 1.0, 'less': 1.0, 'two-sided': 1.0}
        else:
            # If E==0 but O>0 this is infinitely surprising under the null => p ~ 0
            return {'greater': 0.0, 'less': 1.0, 'two-sided': 0.0}

    if _HAS_SCIPY:
        # P(X >= O) = sf(O-1), P(X <= O) = cdf(O)
        p_greater = poisson.sf(O - 1, E)   # P(X >= O)
        p_less = poisson.cdf(O, E)         # P(X <= O)
    else:
        # Fallback iterative computation of lower tail P(X <= O)
        # Start with p0 = exp(-E)
        p0 = math.exp(-E)
        p = p0
        cdf = p0
        for k in range(1, O + 1):
            p = p * (E / k)
            cdf += p
            # no early exit to keep accuracy for moderate O
        p_less = min(max(cdf, 0.0), 1.0)
        if O == 0:
            p_greater = 1.0
        else:
            # P(X >= O) = 1 - P(X <= O-1)
            p_greater = 1.0 - (cdf - p)  # subtract p(O) to get P(X <= O-1)
        p_greater = min(max(p_greater, 0.0), 1.0)

    p_two = 2.0 * min(p_greater, p_less)
    if p_two > 1.0:
        p_two = 1.0

    return {'greater': float(p_greater), 'less': float(p_less), 'two-sided': float(p_two)}


def _benjamini_hochberg(pvals: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Return BH-adjusted pvals and significance mask at level alpha."""
    m = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]
    adjusted = np.empty(m, dtype=float)
    # compute adjusted values (step-up)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1  # 1-based
        raw = (m / rank) * sorted_p[i]
        prev = min(prev, raw)
        adjusted[i] = prev
    # reorder to original order
    adj_orig_order = np.empty(m, dtype=float)
    adj_orig_order[sorted_idx] = adjusted
    signif_mask = (adj_orig_order <= alpha).tolist()
    return adj_orig_order.tolist(), signif_mask


def multiple_test_poisson(
    O: Sequence[int],
    E: Sequence[float],
    alpha: float = 0.05,
    method: str = "fdr_bh",
    sided: str = "two",
    surrogate_max: bool = False,
    n_surrogates: int = 2000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Perform per-bin Poisson tests and multiple-test correction.

    Parameters
    ----------
    O : sequence of ints
        Observed counts per bin.
    E : sequence of floats
        Expected counts per bin under Poisson null (same length as O).
    alpha : float
        Desired family-wise (Bonferroni) or FDR level.
    method : {'bonferroni', 'fdr_bh'}
        Multiple-testing correction for per-bin p-values.
    sided : {'two', 'greater', 'less'}
        'greater' tests for excess (one-sided), 'less' for deficit (one-sided), 'two' for two-sided.
    surrogate_max : bool
        If True, compute a surrogate-based max-statistic global p-value in addition to corrected per-bin tests.
        Surrogates are drawn as independent Poisson(E) realizations (simple null). Set to False to skip.
    n_surrogates : int
        Number of surrogate realizations for the max-statistic test (if surrogate_max=True).
    random_seed : int
        Random seed for surrogate sampling.

    Returns
    -------
    A dict with keys:
      - 'pvals' : list of per-bin p-values (used in correction)
      - 'adj_pvals' : adjusted p-values (BH or Bonferroni-adjusted values)
      - 'significant_mask' : boolean list marking significant bins after correction
      - 'global_significant' : bool, True if any bin significant after correction (FWER/FDR)
      - 'alpha_used' : threshold used (Bonferroni threshold or alpha for FDR)
      - if surrogate_max True:
          - 'surrogate_global_p' : family-wise p-value from max-statistic surrogates
          - 'surrogate_threshold' : threshold on absolute z-statistic corresponding to (1-alpha) quantile of surrogates
    """
    if len(O) != len(E):
        raise ValueError("O and E must have the same length")
    M = len(O)
    if M == 0:
        raise ValueError("Empty input")
    if method not in ("bonferroni", "fdr_bh"):
        raise ValueError("method must be 'bonferroni' or 'fdr_bh'")
    if sided not in ("two", "greater", "less"):
        raise ValueError("sided must be 'two', 'greater', or 'less'")

    O_arr = np.asarray(O, dtype=int)
    E_arr = np.asarray(E, dtype=float)
    # per-bin p-values
    pvals = []
    details = []
    for oi, ei in zip(O_arr, E_arr):
        p = _poisson_pvals_single(int(oi), float(ei))
        if sided == "two":
            pv = p["two-sided"]
        elif sided == "greater":
            pv = p["greater"]
        else:
            pv = p["less"]
        pvals.append(pv)
        details.append(p)

    # multiple-test correction
    if method == "bonferroni":
        threshold = alpha / M
        signif_mask = [pv <= threshold for pv in pvals]
        adj_pvals = [min(pv * M, 1.0) for pv in pvals]
        global_sig = any(signif_mask)
        alpha_used = threshold
    else:  # FDR BH
        adj_pvals, signif_mask = _benjamini_hochberg(pvals, alpha=alpha)
        global_sig = any(signif_mask)
        alpha_used = alpha

    result = {
        "pvals": pvals,
        "adj_pvals": adj_pvals,
        "significant_mask": signif_mask,
        "global_significant": bool(global_sig),
        "alpha_used": alpha_used,
        "details": details
    }

    # surrogate max-statistic (optional): controls FWER under correlated bins if the null is Poisson(E)
    if surrogate_max:
        rng = np.random.RandomState(random_seed)
        # observed z-statistics (use sqrt(E) denom; if E_i==0 set z to 0 for stability)
        z_obs = np.where(E_arr > 0, (O_arr - E_arr) / np.sqrt(E_arr), 0.0)
        M_obs = np.max(np.abs(z_obs))
        surrogate_max_stats = np.empty(n_surrogates, dtype=float)
        for s in range(n_surrogates):
            # draw surrogate counts independently per bin under Pois(E)
            if _HAS_SCIPY:
                Xs = poisson.rvs(E_arr, random_state=rng)
            else:
                # numpy Poisson is fine for surrogates
                Xs = rng.poisson(lam=E_arr)
            zs = np.where(E_arr > 0, (Xs - E_arr) / np.sqrt(E_arr), 0.0)
            surrogate_max_stats[s] = np.max(np.abs(zs))
        # compute empirical p-value (add-one correction)
        surrogate_global_p = (np.sum(surrogate_max_stats >= M_obs) + 1.0) / (n_surrogates + 1.0)
        # threshold on |z| for FWER control:
        surrogate_threshold = np.quantile(surrogate_max_stats, 1.0 - alpha)
        result["surrogate_global_p"] = float(surrogate_global_p)
        result["surrogate_threshold"] = float(surrogate_threshold)
        # if surrogate_global_p <= alpha then reject global null
        result["surrogate_global_significant"] = surrogate_global_p <= alpha

    return result



def scale_for_fraction(obs, pred, fraction=0.8, max_iter=200, tol=1e-9,
                       return_idxs=True, allow_bruteforce=False):
    """
    Compute a scale factor s so that s * pred matches obs closely for only
    `fraction` of the bins (trim the worst (1-fraction) portion).
    
    Minimizes sum_{i in J} (obs_i - s * pred_i)^2 over subsets J of size k = floor(fraction * n).
    Uses an iterative trimmed-least-squares heuristic:
      - initialize s by ordinary least squares on all bins
      - repeatedly pick k bins with smallest residuals and recompute s on them
      - stop when s and the selected subset stabilize or max_iter reached
    
    Arguments:
      obs            : 1-D array-like of observed bin heights (length n)
      pred           : 1-D array-like of predicted/bin heights to be scaled (length n)
      fraction       : fraction of bins to fit (default 0.8)
      max_iter       : maximum iterations for iterative solver (default 200)
      tol            : tolerance for change in s to declare convergence (default 1e-9)
      return_idxs    : whether to return the indices of the bins used in final fit
      allow_bruteforce: if True and n <= 20, try exact combinatorial search for global optimum
                       (warn: exponential; only recommended for n <= 20)
    
    Returns:
      s              : scale factor (float). If cannot be computed (e.g. pred all zeros), returns np.nan
      selected_idx   : (optional) numpy array of indices used in final fit (length k)
      info           : dict with keys:
                        - 'converged' : bool
                        - 'iterations': int
                        - 'k'         : number of bins used (floor(fraction*n))
                        - 'final_rss' : sum of squared residuals on final subset
                        
    Notes:
      - If pred contains zeros, those entries are still considered; they contribute nothing to denominator.
      - If denominator (sum pred^2 on subset) is zero, scale is undefined -> returns NaN.
      - The iterative heuristic usually converges quickly and gives a robust solution.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if obs.shape != pred.shape:
        raise ValueError("obs and pred must have the same shape.")
    n = obs.size
    if n == 0:
        return (np.nan, np.array([], dtype=int), {'converged': True, 'iterations': 0, 'k': 0, 'final_rss': 0}) \
               if return_idxs else (np.nan, {'converged': True, 'iterations': 0, 'k': 0, 'final_rss': 0})

    k = max(1, int(np.floor(fraction * n)))  # at least 1

    # Helper to compute OLS scale for an index set
    def ols_scale(idx):
        x = pred[idx]
        y = obs[idx]
        denom = np.dot(x, x)
        if denom == 0.0:
            return np.nan
        return float(np.dot(y, x) / denom)

    # Optional exact search for small n
    if allow_bruteforce and n <= 20:
        from itertools import combinations
        best_rss = np.inf
        best_s = np.nan
        best_idx = None
        all_idx = np.arange(n)
        for comb in combinations(all_idx, k):
            comb = np.array(comb, dtype=int)
            s_try = ols_scale(comb)
            if np.isnan(s_try):
                continue
            res = obs[comb] - s_try * pred[comb]
            rss = float(np.dot(res, res))
            if rss < best_rss:
                best_rss = rss
                best_s = s_try
                best_idx = comb
        info = {'converged': True, 'iterations': 0, 'k': k, 'final_rss': best_rss}
        if return_idxs:
            return best_s, best_idx, info
        return best_s, info

    # Iterative trimmed least-squares
    # initial OLS on all bins (guard denom)
    denom_all = np.dot(pred, pred)
    if denom_all == 0.0:
        # predicted vector is zero -> no finite scale
        info = {'converged': True, 'iterations': 0, 'k': k, 'final_rss': np.dot(obs, obs)}
        if return_idxs:
            return np.nan, np.arange(n, dtype=int), info
        return np.nan, info
    s = float(np.dot(obs, pred) / denom_all)

    prev_idx = None
    converged = False
    for it in range(1, max_iter + 1):
        res = obs - s * pred
        sqr = res * res
        # keep k bins with smallest squared residuals
        idx = np.argpartition(sqr, k - 1)[:k] if k < n else np.arange(n)
        # sort indices by residual ascending (so deterministic)
        idx = idx[np.argsort(sqr[idx])]
        s_new = ols_scale(idx)
        # if denominator was zero (all pred[idx]==0) s_new is nan; keep s and treat as converged
        if np.isnan(s_new):
            info = {'converged': False, 'iterations': it, 'k': k, 'final_rss': float(np.dot(obs[idx] - s * pred[idx], obs[idx] - s * pred[idx]))}
            if return_idxs:
                return s, idx, info
            return s, info
        # check convergence (s and selected indices)
        if abs(s_new - s) <= tol and (prev_idx is not None and np.array_equal(np.sort(prev_idx), np.sort(idx))):
            s = s_new
            converged = True
            break
        s = s_new
        prev_idx = idx.copy()
    # final RSS on selected subset
    final_res = obs[idx] - s * pred[idx]
    final_rss = float(np.dot(final_res, final_res))
    info = {'converged': converged, 'iterations': it, 'k': k, 'final_rss': final_rss}
    if return_idxs:
        return s, idx, info
    return s, info
