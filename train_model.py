import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
from math import factorial
from utils import load_data, compute_team_strengths, elo_from_stats, compute_recent_form
from datetime import timedelta


def dc_adjustment(h,g):
    return {
        (0,0): lambda rho: 1 - rho,
        (0,1): lambda rho: 1 + rho,
        (1,0): lambda rho: 1 + rho,
        (1,1): lambda rho: 1 - rho
    }.get((h,g), lambda rho: 1)


def joint_prob_dc(mu1, mu2, rho, max_goals=6):
    p1 = [np.exp(-mu1) * mu1**i / factorial(i) for i in range(max_goals+1)]
    p2 = [np.exp(-mu2) * mu2**j / factorial(j) for j in range(max_goals+1)]
    P = np.outer(p1,p2)
    P_adj = P.copy()
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            adj = dc_adjustment(i,j)(rho)
            P_adj[i,j] *= adj
    P_adj /= P_adj.sum()
    return P_adj


def neg_log_likelihood(params, data, teams_index, decay_half_life_days=180):
    n = len(teams_index)
    la = params[:n]
    ld = params[n:2*n]
    home_adv = params[-2]
    rho = params[-1]
    attacks = {team: np.exp(la[i]) for team,i in teams_index.items()}
    defenses = {team: np.exp(ld[i]) for team,i in teams_index.items()}
    ll = 0.0
    now = data['date'].max()
    for _, r in data.iterrows():
        h = r['home_team']; a = r['away_team']
        mu_h = attacks[h] * defenses[a] * home_adv
        mu_a = attacks[a] * defenses[h]
        maxg = max(int(r['home_goals']), int(r['away_goals']), 6)
        P = joint_prob_dc(mu_h, mu_a, rho, max_goals=maxg)
        obs_prob = P[int(r['home_goals']), int(r['away_goals'])]
        days = (now - r['date']).days
        weight = 2 ** (-days/decay_half_life_days)
        ll += weight * np.log(max(1e-15, obs_prob))
    return -ll


def train(matches_path, stats_path, output_model_path='dc_model.pkl'):
    matches, stats = load_data(matches_path, stats_path)
    teams = sorted(pd.unique(matches[['home_team','away_team']].values.ravel('K')))
    teams_index = {t:i for i,t in enumerate(teams)}
    attacks, defenses, home_adv = compute_team_strengths(matches, teams, stats)
    n = len(teams)
    la0 = np.log(np.array([attacks[t] for t in teams]))
    ld0 = np.log(np.array([defenses[t] for t in teams]))
    rho0 = 0.05
    x0 = np.concatenate([la0, ld0, np.array([home_adv, rho0])])
    bnds = [(-5,5)]*(2*n) + [(0.5,2.5),(-0.9,0.9)]
    res = minimize(neg_log_likelihood, x0, args=(matches, teams_index),
                   method='L-BFGS-B', bounds=bnds, options={'maxiter':200})
    params = res.x
    la = params[:n]; ld = params[n:2*n]; home_adv = params[-2]; rho = params[-1]
    model = {
        'teams': teams,
        'attacks': {t: float(np.exp(la[i])) for t,i in teams_index.items()},
        'defenses': {t: float(np.exp(ld[i])) for t,i in teams_index.items()},
        'home_adv': float(home_adv),
        'rho': float(rho),
        'trained_on_until': matches['date'].max()
    }
    model['elos'] = elo_from_stats(teams, matches, stats)
    model['last5'] = compute_recent_form(matches, teams, as_of_date=model['trained_on_until'] + timedelta(days=1), n=5)
    with open(output_model_path,'wb') as f:
        pickle.dump(model,f)
    return model

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches','-m', required=True)
    parser.add_argument('--stats','-s', required=True)
    parser.add_argument('--out','-o', default='dc_model.pkl')
    args = parser.parse_args()
    train(args.matches, args.stats, args.out)