import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from datetime import timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
from train_model import joint_prob_dc

# ================================
# Cargar modelo entrenado
# ================================
@st.cache_resource
def load_model():
    with open("dc_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# ================================
# Funciones auxiliares
# ================================
def predict_match_probs(home, away, model, max_goals=10):
    mu_h = model['attacks'][home] * model['defenses'][away] * model['home_adv']
    mu_a = model['attacks'][away] * model['defenses'][home]

    P = joint_prob_dc(mu_h, mu_a, model['rho'], max_goals=max_goals)

    home_w = draw = away_w = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if i > j:
                home_w += P[i, j]
            elif i == j:
                draw += P[i, j]
            else:
                away_w += P[i, j]

    # Top marcadores
    pairs = [((i, j), P[i, j]) for i in range(P.shape[0]) for j in range(P.shape[1])]
    pairs_sorted = sorted(pairs, key=lambda x: -x[1])[:10]
    top_scores = [
        {"score": f"{i}-{j}", "prob": f"{p*100:.2f}%"}
        for (i, j), p in pairs_sorted[:5]
    ]

    return {
        "home": home_w,
        "draw": draw,
        "away": away_w,
        "top_scores": top_scores
    }

def df_to_pdf(df):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 10)
    for line in df.to_string(index=False).split("\n"):
        text.textLine(line)
    c.drawText(text)
    c.save()
    buffer.seek(0)
    return buffer

def df_to_jpg(df):
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.4 + 1))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", dpi = 600, bbox_inches="tight")
    buf.seek(0)
    return buf


# ================================
# Barra lateral
# ================================

st.sidebar.title("Menú de Navegación")
page = st.sidebar.radio("Ir a:", [
    "Pagina Principal",
    "Predicción de un partido",
    "Predicción múltiple",
    "Monte Carlo",
    "Ranking ELOs",
    "Tabla de Posiciones"
])


#############

#Pagina Principal

if page == "Pagina Principal":  

    st.set_page_config(layout='wide')
    st.title('Predictor de resultados de la Primera Division del Futbol Argentino')
    st.write("Modelo predictivo de resultados del futbol argentino. Hay cuatro paginas. La primera es para hacer predicciones individuales. La segunda para realizar varios pronosticos al mismo tiempo. La tercera es para hacer una simulacion Montecarlo. La cuarta pagina muestra la tabla de equipos, junto a su ranking calculado por ELO. En la ultima se visualiza la tabla de posiciones actual del futbol argentino")


# ================================
# Página: Predicción de un partido
# ================================
elif page == "Predicción de un partido":
    st.header("Predicción de un partido")

    equipos = model["teams"]

    home = st.selectbox("Equipo Local", equipos, index = 0)
    away = st.selectbox("Equipo Visitante", equipos, index = 1)

    if st.button("Predecir"):
        res = predict_match_probs(home, away, model)

        st.subheader('Probabilidades de resultado')
        col1,col2,col3 = st.columns(3)
        col1.metric(f'{home} gana', f"{res['home']*100:.2f}%")
        col2.metric('Empate', f"{res['draw']*100:.2f}%")
        col3.metric(f'{away} gana', f"{res['away']*100:.2f}%")

        st.subheader('Marcadores más probables')
        st.table(pd.DataFrame(res["top_scores"]))

# ================================
# Página: Predicción múltiple
# ================================
elif page == "Predicción múltiple":
    st.header("Predicción de varios partidos")

    option = st.radio("¿Cómo querés cargar los partidos?", ["Subir CSV", "Cargar manualmente"])

    if option == "Subir CSV":
        uploaded_file = st.file_uploader("Subí un CSV con columnas home, away", type="csv")

        if uploaded_file is not None:
            matches_batch = pd.read_csv(uploaded_file)

    elif option == "Cargar manualmente":
        num_matches = st.number_input("Cantidad de partidos", min_value=1, max_value=15, value=3, step=1)
        matches_manual = []
        equipos = model["teams"]

        for i in range(num_matches):
            st.subheader(f"Partido {i+1}")
            home = st.selectbox(f"Equipo Local {i+1}", equipos, key=f"home_{i}")
            away = st.selectbox(f"Equipo Visitante {i+1}", equipos, key=f"away_{i}")
            matches_manual.append({"home": home, "away": away})

        matches_batch = pd.DataFrame(matches_manual)

    # Si tenemos partidos cargados, mostramos resultados
    if "matches_batch" in locals():
        if not matches_batch.empty:
            results = []
            for _, row in matches_batch.iterrows():
                res = predict_match_probs(row['home'], row['away'], model)
                results.append({
                    "Partido": f"{row['home']} vs {row['away']}",
                    "Prob. Local (%)": f"{res['home']*100:.1f}%",
                    "Prob. Empate (%)": f"{res['draw']*100:.1f}%",
                    "Prob. Visita (%)": f"{res['away']*100:.1f}%",
                    "Top marcadores": ", ".join([f"{s['score']} ({s['prob']})" for s in res["top_scores"]])
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Descargar en Excel
            towrite = io.BytesIO()
            results_df.to_excel(towrite, index=False, sheet_name="Predicciones")
            towrite.seek(0)
            st.download_button(
                label="📊 Descargar en Excel",
                data=towrite,
                file_name="predicciones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Descargar en JPG
            st.download_button(
                label="🖼️ Descargar en JPEG",
                data=df_to_jpg(results_df),
                file_name="predicciones.jpg",
                mime="image/jpeg"
            )


# ================================
# Página: Monte Carlo
# ================================
elif page == "Monte Carlo":
    st.header("Simulación Monte Carlo")

    home = st.selectbox("Equipo Local", model["teams"], key="mc_home",index = 0)
    away = st.selectbox("Equipo Visitante", model["teams"], key="mc_away",index = 1)

    mc_runs = st.slider("Número de simulaciones", 1000, 50000, 10000, step=1000)

    if st.button("Simular"):
        # Ejemplo de simulación
        home_goals = np.random.poisson(1.5, mc_runs)
        away_goals = np.random.poisson(1.2, mc_runs)

        home_win = np.mean(home_goals > away_goals) * 100
        draw = np.mean(home_goals == away_goals) * 100
        away_win = np.mean(home_goals < away_goals) * 100

        st.success(f"**{home}** gana: {home_win:.1f}%")
        st.info(f"Empate: {draw:.1f}%")
        st.warning(f"**{away}** gana: {away_win:.1f}%")

# ================================
# Página: Ranking ELOs
# ================================
elif page == "Ranking ELOs":
    st.header("Ranking ELOs por equipo")

    elo_df = pd.DataFrame(model["elos"].items(), columns=["Equipo", "ELO"])
    elo_df = elo_df.sort_values("ELO", ascending=False).reset_index(drop=True)
    elo_df.index = elo_df.index + 1
    elo_df.index.name = "Ranking"
    st.table(elo_df)
    
# ================================
# Página: Tabla Posiciones
# ================================

elif page == "Tabla de Posiciones":

        st.components.v1.iframe(
        "https://widgets.sofascore.com/embed/tournament/156467/season/77826/standings/Group%20A?widgetTitle=Group%20A&showCompetitionLogo=true",
        height=923,
        width=900,
        scrolling=True
        )
        st.components.v1.iframe(
        "https://widgets.sofascore.com/embed/tournament/156467/season/77826/standings/Group%20B?widgetTitle=Group%20B&showCompetitionLogo=true",
        height=923,
        width=900,
        scrolling=True
        )
       
