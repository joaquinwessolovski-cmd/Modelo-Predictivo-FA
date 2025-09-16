# Project: Poisson Bivariate + Dixon-Coles model for Argentine football
# Files provided in this single code document (split by headers):
# 1) utils.py            -> helpers to load data, compute features, ELO, form
# 2) train_model.py      -> trains the model, saves model parameters to a .pkl
# 3) predict_app.py      -> Streamlit app to load model and produce predictions + MonteCarlo
# 4) requirements.txt    -> dependencies
#
# Save each section below into separate files with the indicated filenames.

############################## utils.py ##############################
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from datetime import timedelta

def load_data(matches_path: str, stats_path: str):
    matches = pd.read_excel(matches_path, engine='openpyxl')
    stats = pd.read_excel(stats_path, engine='openpyxl')
    matches['date'] = pd.to_datetime(matches['date'])
    return matches, stats


def compute_recent_form(matches: pd.DataFrame, teams: list, as_of_date, n=5):
    df = matches[matches['date'] < as_of_date].sort_values('date')
    out = {t: {'overall_pts':0,'home_pts':0,'away_pts':0,'games_overall':0,'games_home':0,'games_away':0}
           for t in teams}
    for t in teams:
        played = df[(df['home_team']==t)|(df['away_team']==t)].sort_values('date',ascending=False).head(n)
        pts=0; hpts=0; apt=0
        go=0; gh=0; ga=0
        for _,r in played.iterrows():
            go+=1
            if r['home_team']==t:
                gh+=1
                if r['home_goals']>r['away_goals']:
                    pts+=3; hpts+=3
                elif r['home_goals']==r['away_goals']:
                    pts+=1; hpts+=1
            else:
                ga+=1
                if r['away_goals']>r['home_goals']:
                    pts+=3; apt+=3
                elif r['away_goals']==r['home_goals']:
                    pts+=1; apt+=1
        out[t]['overall_pts']=pts
        out[t]['home_pts']=hpts
        out[t]['away_pts']=apt
        out[t]['games_overall']=go
        out[t]['games_home']=gh
        out[t]['games_away']=ga
    return out


def compute_team_strengths(matches, teams, stats, decay_half_life_days=180, recent_matches=5):
    """
    Calcula la fuerza ofensiva, defensiva y ventaja de local de cada equipo.
    Pondera la forma reciente usando los últimos 'recent_matches' partidos y decay temporal.
    
    matches: DataFrame con columnas ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
    teams: lista de nombres de equipos
    stats: DataFrame con columnas ['team', 'xG', 'xGA', 'xG_home', 'xGA_home', 'xG_away', 'xGA_away']
    decay_half_life_days: días de mitad de vida para decay temporal
    recent_matches: cantidad de últimos partidos a considerar
    """

    attacks = {}
    defenses = {}
    home_adv = {}

    # Convertir fechas de partidos a datetime si no lo están
    matches['date'] = pd.to_datetime(matches['date'])

    now = matches['date'].max()

    for t in teams:
        # Filtrar partidos recientes del equipo
        team_matches = matches[(matches['home_team'] == t) | (matches['away_team'] == t)].sort_values('date', ascending=False)
        team_matches_recent = team_matches.head(recent_matches)

        # Decay temporal: más reciente = más peso
        days_diff = (now - team_matches_recent['date']).dt.days
        weights_time = 2 ** (-days_diff / decay_half_life_days)

        # Calcular xG ofensivo y defensivo usando stats
        team_stats = stats[stats['team'] == t]

        if not team_stats.empty:
            xg = np.average(team_stats['xG'], weights=np.ones(len(team_stats)))
            xga = np.average(team_stats['xGA'], weights=np.ones(len(team_stats)))
            xg_home = np.average(team_stats['xG_home'], weights=np.ones(len(team_stats)))
            xg_away = np.average(team_stats['xG_away'], weights=np.ones(len(team_stats)))
            xga_home = np.average(team_stats['xGA_home'], weights=np.ones(len(team_stats)))
            xga_away = np.average(team_stats['xGA_away'], weights=np.ones(len(team_stats)))
        else:
            xg = xga = xg_home = xg_away = xga_home = xga_away = 1.0  # valores por defecto si no hay stats

        # Incorporar forma reciente (promedio goles últimos partidos)
        recent_goals_scored = (
            team_matches_recent.loc[team_matches_recent['home_team'] == t, 'home_goals'].mean()
            + team_matches_recent.loc[team_matches_recent['away_team'] == t, 'away_goals'].mean()
        )

        recent_goals_conceded = (
            team_matches_recent.loc[team_matches_recent['home_team'] == t, 'away_goals'].mean()
            + team_matches_recent.loc[team_matches_recent['away_team'] == t, 'home_goals'].mean()
        )

        # Ponderar xG y xGA con forma reciente (peso 50%)
        xg_final = 0.3 * xg + 0.7 * recent_goals_scored
        xga_final = 0.3 * xga + 0.7 * recent_goals_conceded

        attacks[t] = xg_final
        defenses[t] = xga_final
        home_adv = 0.1  # valor fijo por defecto, podés ajustarlo
    return attacks, defenses, float(home_adv)

def elo_from_stats(teams, matches, stats_df, k=20, start_elo=1500):
    elos = {t: start_elo for t in teams}
    matches_sorted = matches.sort_values('date')
    for _,r in matches_sorted.iterrows():
        h = r['home_team']; a = r['away_team']
        hg = r['home_goals']; ag = r['away_goals']
        # Use xG if available
        h_stats = stats_df[stats_df['team']==h]
        a_stats = stats_df[stats_df['team']==a]
        if not h_stats.empty and not a_stats.empty:
            hg_eff = 0.7*hg + 0.3*h_stats['xG'].values[0]
            ag_eff = 0.7*ag + 0.3*a_stats['xG'].values[0]
        else:
            hg_eff, ag_eff = hg, ag
        diff = elos[h]-elos[a]
        exp_h = 1/(1+10**(-diff/400))
        if hg_eff>ag_eff: actual_h=1
        elif hg_eff==ag_eff: actual_h=0.5
        else: actual_h=0
        m = abs(hg_eff-ag_eff)
        K = k * (1 + np.log1p(m))
        elos[h] += K*(actual_h - exp_h)
        elos[a] += K*((1-actual_h) - (1-exp_h))
    return elos
