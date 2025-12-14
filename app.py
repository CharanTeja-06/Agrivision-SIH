import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="AgriVision • AI Crop Yield & Fertilizer Optimizer",
    page_icon="🌾",
    layout="wide"
)

# =========================================================
# Global custom CSS
# =========================================================
st.markdown(
    """
    <style>
    body {
        margin: 0;
        padding: 0;
        background: radial-gradient(circle at top, #1d283a 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        margin-bottom: 1.5rem;
    }
    @media (max-width: 900px) {
        .hero { flex-direction: column; }
    }
    .hero-left { max-width: 640px; }
    .hero-title {
        font-size: 2.2rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-align: left;
        word-wrap: break-word;
}
    .hero-subtitle {
        color: #9ca3af;
        font-size: 0.98rem;
    }
    .hero-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.9rem;
    }
    .hero-badge {
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        border: 1px solid rgba(148,163,184,0.6);
        color: #e5e7eb;
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.3));
        backdrop-filter: blur(12px);
    }

    .hero-right {
        flex: 1;
        display: flex;
        justify-content: center;
    }
    .hero-card {
        width: 100%;
        max-width: 420px;
        position: relative;
        padding: 1.3rem 1.4rem;
        border-radius: 22px;
        background: radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 55%),
                    radial-gradient(circle at bottom right, rgba(168,85,247,0.12), transparent 55%),
                    rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.4);
        box-shadow: 0 24px 60px rgba(15,23,42,0.9);
        backdrop-filter: blur(18px);
    }
    .hero-stat-main {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .hero-stat-sub {
        font-size: 0.8rem;
        color: #9ca3af;
    }

    .glass-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(15,23,42,0.7));
        border-radius: 20px;
        border: 1px solid rgba(148,163,184,0.4);
        box-shadow: 0 20px 50px rgba(15,23,42,0.9);
        padding: 1.6rem 1.8rem;
        backdrop-filter: blur(16px);
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .section-subtitle {
        font-size: 0.86rem;
        color: #9ca3af;
        margin-bottom: 1.0rem;
    }

    .metric-card {
        background: radial-gradient(circle at top, #22c55e, #15803d);
        border-radius: 18px;
        padding: 1.8rem 1.5rem;
        color: white;
        box-shadow: 0 24px 80px rgba(22,163,74,0.85);
        text-align: left;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(255,255,255,0.32), transparent 70%);
        top: -40px;
        right: -60px;
        opacity: 0.3;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        opacity: 0.9;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }
    .metric-sub { font-size: 0.9rem; opacity: 0.95; }

    .npk-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    .npk-low {
        background: rgba(185,28,28,0.09);
        border: 1px solid rgba(248,113,113,0.9);
        color: #fecaca;
    }
    .npk-ok {
        background: rgba(22,163,74,0.14);
        border: 1px solid rgba(74,222,128,0.9);
        color: #bbf7d0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Hero section
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-left">
            <div class="hero-title">
                AgriVision: AI Crop Yield & Fertilizer Optimizer
            </div>
            <div class="hero-subtitle">
                An AI‑driven assistant for Indian agriculture that predicts crop yield and
                intelligently suggests NPK fertiliser doses in a single, beautiful dashboard.
            </div>
            <div class="hero-badge-row">
                <div class="hero-badge">AI Crop Yield</div>
                <div class="hero-badge">NPK Optimizer</div>
                <div class="hero-badge">Streamlit • Python</div>
            </div>
        </div>
        <div class="hero-right">
            <div class="hero-card">
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.16em; color:#9ca3af;">
                    AI Insight Snapshot
                </div>
                <div style="margin-top:0.6rem;">
                    <div class="hero-stat-main">🌾 Yield • Fertiliser • Planning</div>
                    <div class="hero-stat-sub">
                        Blend historical yield data with intelligent nutrient balancing to move from guesswork
                        to AI‑assisted farm decisions.
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Data loading & model training (no Year, no District)
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(path: str = "crop_yield.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    # Expect at least State, Crop, Season, Area, and either Yield or Production.[web:16]
    if "Yield" not in df.columns:
        if {"Production", "Area"}.issubset(df.columns):
            df["Yield"] = df["Production"] / df["Area"]
        else:
            raise ValueError("Dataset must contain either 'Yield' or both 'Production' and 'Area'.")
    return df


@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):
    required_cols = ["State", "Crop", "Season", "Area", "Yield"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    df = df.copy()

    cat_cols = ["State", "Crop", "Season"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = ["State", "Crop", "Season", "Area"]
    X = df[feature_cols]
    y = df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    return model, encoders, feature_cols, r2


@st.cache_data(show_spinner=False)
def get_unique_options(df: pd.DataFrame):
    states = sorted(df["State"].astype(str).unique().tolist())
    crops = sorted(df["Crop"].astype(str).unique().tolist())
    seasons = sorted(df["Season"].astype(str).unique().tolist())
    return states, crops, seasons


# Fertilizer reference values
CROP_NPK_REQUIREMENTS = {
    "Rice":      {"N": 120, "P": 60,  "K": 40},
    "Wheat":     {"N": 120, "P": 60,  "K": 40},
    "Maize":     {"N": 150, "P": 60,  "K": 40},
    "Cotton":    {"N": 120, "P": 60,  "K": 60},
    "Sugarcane": {"N": 250, "P": 115, "K": 115},
}
FERTILIZER_CONTENT = {
    "Urea": {"N": 0.46, "P": 0.0,  "K": 0.0},
    "DAP":  {"N": 0.18, "P": 0.46, "K": 0.0},
    "MOP":  {"N": 0.0,  "P": 0.0,  "K": 0.60},
}
BAG_SIZE_KG = 50


def calculate_fertilizer_dose(crop: str, soil_n: float, soil_p: float, soil_k: float, area_ha: float = 1.0):
    if crop not in CROP_NPK_REQUIREMENTS:
        raise ValueError(f"NPK standard not defined for crop: {crop}")
    req = CROP_NPK_REQUIREMENTS[crop]

    req_n_total = req["N"] * area_ha
    req_p_total = req["P"] * area_ha
    req_k_total = req["K"] * area_ha

    avail_n_total = soil_n * area_ha
    avail_p_total = soil_p * area_ha
    avail_k_total = soil_k * area_ha

    deficit_n = max(req_n_total - avail_n_total, 0)
    deficit_p = max(req_p_total - avail_p_total, 0)
    deficit_k = max(req_k_total - avail_k_total, 0)

    urea_kg = deficit_n / FERTILIZER_CONTENT["Urea"]["N"] if deficit_n > 0 else 0
    dap_kg  = deficit_p / FERTILIZER_CONTENT["DAP"]["P"]  if deficit_p > 0 else 0
    mop_kg  = deficit_k / FERTILIZER_CONTENT["MOP"]["K"]  if deficit_k > 0 else 0

    return {
        "requirements_per_ha": req,
        "deficit_per_ha": {
            "N": deficit_n / max(area_ha, 1e-6),
            "P": deficit_p / max(area_ha, 1e-6),
            "K": deficit_k / max(area_ha, 1e-6),
        },
        "fertilizer_kg": {
            "Urea": urea_kg,
            "DAP": dap_kg,
            "MOP": mop_kg,
        },
        "fertilizer_bags": {
            "Urea": urea_kg / BAG_SIZE_KG,
            "DAP": dap_kg / BAG_SIZE_KG,
            "MOP": mop_kg / BAG_SIZE_KG,
        }
    }


# Load & train
try:
    data = load_data("crop_yield.csv")
    model, encoders, feature_cols, r2_score = train_model(data)
    states, crops, seasons = get_unique_options(data)
except Exception as e:
    st.error(f"Error initializing model or data: {e}")
    st.stop()

# =========================================================
# Tabs as pages: Overview, Yield, Fertilizer
# =========================================================
tab_overview, tab_yield, tab_fert = st.tabs(
    ["🤖 AI Overview", "🌱 Yield Predictor", "🧪 Fertilizer Optimizer"]
)

# ---------------------- Overview ----------------------
with tab_overview:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">AI‑Powered Farm Intelligence</div>
        <div class="section-subtitle">
            Behind the scenes, AgriVision blends historical crop yield data with machine learning
            to estimate productivity and guide NPK decisions for Indian farms.
        </div>
        """,
        unsafe_allow_html=True
    )
    c1, c2 = st.columns([1.2, 1.4])
    with c1:
        st.write("**Rows in dataset:**", len(data))
        st.write("**States covered:**", len(states))
        st.write("**Unique crops:**", len(crops))
        st.write("**Seasons:**", len(seasons))
    with c2:
        st.markdown(
            """
            - The **Yield Predictor** uses a Random Forest model trained on your dataset features.  
            - The **Fertilizer Optimizer** compares soil N‑P‑K with typical crop requirements and 
              converts the gap into Urea, DAP and MOP doses.  
            - You can plug richer datasets later (weather, soil, remote sensing) into the same UI.
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Yield Predictor ----------------------
with tab_yield:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">AI Yield Prediction</div>
        <div class="section-subtitle">
            Select your state, crop, season and area to estimate yield (q/ha) and total production.
        </div>
        """,
        unsafe_allow_html=True
    )

    c_form, c_metric = st.columns([1.6, 1.4])
    with c_form:
        col1, col2 = st.columns(2)
        with col1:
            ui_state = st.selectbox("State", states)
            ui_season = st.selectbox("Season", seasons)
        with col2:
            ui_crop = st.selectbox("Crop", crops)

        ui_area = st.number_input(
            "Area under cultivation (hectares)",
            min_value=0.1,
            max_value=10000.0,
            value=float(max(data["Area"].median(), 1.0)),
            step=0.1
        )

        predict_btn = st.button("🔮 Run Yield Prediction", use_container_width=True)

    with c_metric:
        st.markdown(
            f"**Model validation R²:** `{r2_score:.2f}`  (features: State, Crop, Season, Area)"
        )

    if predict_btn:
        try:
            e_state = encoders["State"].transform([ui_state])[0]
            e_crop = encoders["Crop"].transform([ui_crop])[0]
            e_season = encoders["Season"].transform([ui_season])[0]
        except Exception:
            st.error("The selected values are not recognized by the encoders. Check dataset categories.")
        else:
            row = pd.DataFrame(
                [[e_state, e_crop, e_season, ui_area]],
                columns=feature_cols
            )
            pred_yield = model.predict(row)[0]
            total_prod = pred_yield * ui_area

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Yield</div>
                    <div class="metric-value">{pred_yield:.2f} q/ha</div>
                    <div class="metric-sub">
                        For <b>{ui_area:.2f} ha</b>, estimated total production is
                        <b>{total_prod:.2f} quintals</b>.  
                        Use this as an AI assistant for planning, alongside local expert guidance.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Fertilizer Optimizer ----------------------
CROP_LIST_FOR_FERT = list(CROP_NPK_REQUIREMENTS.keys())

with tab_fert:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">NPK Fertilizer Optimizer</div>
        <div class="section-subtitle">
            Compare your soil test values with crop-specific NPK needs and get predicted Urea, DAP and MOP doses.
        </div>
        """,
        unsafe_allow_html=True
    )

    c_form, c_summary = st.columns([1.4, 1.6])
    with c_form:
        fert_crop = st.selectbox("Crop", CROP_LIST_FOR_FERT)
        fert_area = st.number_input(
            "Field size (hectares)",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1
        )
        st.markdown("**Soil test values (approx. kg/ha)**")
        s_n = st.slider("Nitrogen (N)", 0, 300, 40, 5)
        s_p = st.slider("Phosphorus (P₂O₅)", 0, 200, 20, 5)
        s_k = st.slider("Potassium (K₂O)", 0, 200, 20, 5)
        fert_btn = st.button("🧮 Predict Fertilizer Plan", use_container_width=True)

    with c_summary:
        st.markdown("#### Standard NPK requirements per ha")
        st.dataframe(pd.DataFrame(CROP_NPK_REQUIREMENTS).T.astype(int), use_container_width=True, height=210)

    if fert_btn:
        res = calculate_fertilizer_dose(fert_crop, s_n, s_p, s_k, fert_area)
        req = res["requirements_per_ha"]
        deficit = res["deficit_per_ha"]
        fert_kg = res["fertilizer_kg"]
        fert_bags = res["fertilizer_bags"]

        st.markdown("---")
        c_n, c_p, c_k = st.columns(3)

        def chip(label, d):
            cls = "npk-low" if d > 5 else "npk-ok"
            status = "LOW" if d > 5 else "OK"
            return f'<span class="npk-chip {cls}">{label}: {status}</span>'

        with c_n:
            st.markdown(f"**N** target: {req['N']} kg/ha • Soil: {s_n} kg/ha • Deficit: {deficit['N']:.1f} kg/ha")
            st.markdown(chip("Nitrogen", deficit["N"]), unsafe_allow_html=True)
        with c_p:
            st.markdown(f"**P** target: {req['P']} kg/ha • Soil: {s_p} kg/ha • Deficit: {deficit['P']:.1f} kg/ha")
            st.markdown(chip("Phosphorus", deficit["P"]), unsafe_allow_html=True)
        with c_k:
            st.markdown(f"**K** target: {req['K']} kg/ha • Soil: {s_k} kg/ha • Deficit: {deficit['K']:.1f} kg/ha")
            st.markdown(chip("Potassium", deficit["K"]), unsafe_allow_html=True)

        col_kg, col_bags = st.columns(2)
        with col_kg:
            st.markdown("#### Predicted fertiliser required (kg for field)")
            st.write(
                f"- Urea: **{fert_kg['Urea']:.1f} kg**\n"
                f"- DAP: **{fert_kg['DAP']:.1f} kg**\n"
                f"- MOP: **{fert_kg['MOP']:.1f} kg**"
            )
        with col_bags:
            st.markdown("#### Predicted 50 kg bags")
            st.write(
                f"- Urea: **{fert_bags['Urea']:.1f} bags**\n"
                f"- DAP: **{fert_bags['DAP']:.1f} bags**\n"
                f"- MOP: **{fert_bags['MOP']:.1f} bags**"
            )

        notes = []
        if deficit["N"] > 5:
            notes.append("Nitrogen is low; apply **Urea** or N‑rich complex fertiliser.")
        if deficit["P"] > 5:
            notes.append("Phosphorus is low; **DAP** is a common N + P source.")
        if deficit["K"] > 5:
            notes.append("Potassium is low; use **MOP** or NPK with higher K.")
        if not notes:
            notes.append("Soil NPK levels are near target; maintain with balanced fertiliser and organic matter.")

        st.markdown("#### AI Recommendation")
        for m in notes:
            st.markdown(f"- {m}")

        st.info("Always cross‑check final doses with official state agriculture recommendations or a soil scientist.")
    st.markdown("</div>", unsafe_allow_html=True)
