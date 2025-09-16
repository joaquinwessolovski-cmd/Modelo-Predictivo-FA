import streamlit as st
import pandas as pd
import numpy as np
import pickle
from train_model import joint_prob_dc

st.set_page_config(layout='wide')
st.title('Predictor — Futbol Argentino')
st.write('Modelo predictivo de resultados del futbol argentino. Se eligen los equipos local y visitante, y se muestran las probabilidades de cada resultado, y los 5 marcadores mas probables. Ademas, se puede realizar una simulacion Montecarlo para mayor precision.',size = 10)

@st.cache_data
def load_model(pkl_path):
    with open(pkl_path,'rb') as f:
        return pickle.load(f)

# Load model directly
model = load_model('dc_model.pkl')
teams = model['teams']

home = st.selectbox('Local', teams, index=0)
away = st.selectbox('Visitante', teams, index=1)
max_goals = st.sidebar.slider('Max goles a considerar', 3, 8, 6)
mc_runs = st.sidebar.number_input('Numero de Simulaciones', min_value=100, max_value=20000, value=2000, step=100)

if home == away:
    st.error('El equipo local y visitante deben ser distintos')
    st.stop()

mu_h = model['attacks'][home] * model['defenses'][away] * model['home_adv']
mu_a = model['attacks'][away] * model['defenses'][home]

st.write(f'lambda_local = {mu_h:.3f}, lambda_visitante = {mu_a:.3f}')

P = joint_prob_dc(mu_h, mu_a, model['rho'], max_goals=max_goals)
home_w = draw = away_w = 0.0
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        if i>j: home_w += P[i,j]
        elif i==j: draw += P[i,j]
        else: away_w += P[i,j]

st.subheader('Probabilidades de resultado')
col1,col2,col3 = st.columns(3)
col1.metric(f'{home} gana', f"{home_w*100:.2f}%")
col2.metric('Empate', f"{draw*100:.2f}%")
col3.metric(f'{away} gana', f"{away_w*100:.2f}%")

pairs = [((i,j), P[i,j]) for i in range(P.shape[0]) for j in range(P.shape[1])]
pairs_sorted = sorted(pairs, key=lambda x: -x[1])[:10]
rows = [
    {
        'Marcador': f'{i}-{j}',
        'Probabilidad': f'{p*100:.2f}%'   # formateo como porcentaje
    } 
    for (i,j), p in pairs_sorted[:5]
]
st.subheader('Marcadores más probables')
st.table(pd.DataFrame(rows))

if st.button('Ejecutar Monte Carlo'):
    probs = P.flatten()
    outcomes = [f"{i}-{j}" for i in range(P.shape[0]) for j in range(P.shape[1])]
    sims = np.random.choice(outcomes, size=mc_runs, p=probs)
    from collections import Counter
    c = Counter(sims)
    top = c.most_common(10)
    st.write('Top resultados (MC):')
    st.table(pd.DataFrame([
    {
        'Marcador': t[0],
        'Probabilidad': f"{t[1]/mc_runs*100:.2f}%"
    } 
    for t in top
]))

