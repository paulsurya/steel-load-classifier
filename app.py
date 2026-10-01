import os
import pickle
import tempfile
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Steel_industry_data.csv")
LOCAL_FALLBACK = os.path.join(BASE_DIR, "model_trained.pkl")
FALLBACK_PATH = os.path.join(tempfile.gettempdir(), "model_trained.pkl")

CLASS_LABELS = {1: "Light Load", 2: "Medium Load", 3: "Maximum Load"}

NUMERIC_FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + ["Day_factorized", "WeekStatus_factorized"]

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

FEATURE_INFO = [
    ("Usage_kWh", "Total energy consumed in the interval"),
    ("Lagging_Current_Reactive.Power_kVarh", "Lagging reactive power drawn"),
    ("Leading_Current_Reactive_Power_kVarh", "Leading reactive power drawn"),
    ("CO2(tCO2)", "Carbon dioxide produced"),
    ("Lagging_Current_Power_Factor", "Lagging power factor"),
    ("Leading_Current_Power_Factor", "Leading power factor"),
    ("NSM", "Seconds elapsed since midnight"),
    ("Day_factorized", "Day code (1 = Monday ... 7 = Sunday)"),
    ("WeekStatus_factorized", "Day type code (1 = Weekday, 2 = Weekend)"),
]

FEATURE_SPECS = [
    ("Usage_kWh", 0.0, 160.0, 0.01, 30.0, "%.2f"),
    ("Lagging_Current_Reactive.Power_kVarh", 0.0, 100.0, 0.01, 15.0, "%.2f"),
    ("Leading_Current_Reactive_Power_kVarh", 0.0, 30.0, 0.01, 0.0, "%.2f"),
    ("CO2(tCO2)", 0.0, 0.08, 0.0001, 0.013, "%.4f"),
    ("Lagging_Current_Power_Factor", 0.0, 100.0, 0.01, 85.0, "%.2f"),
    ("Leading_Current_Power_Factor", 0.0, 100.0, 0.01, 100.0, "%.2f"),
]

CSS = """
<style>
    html, body, [class*="css"], .stApp, button, input, select, textarea {
        font-family: Arial, Helvetica, sans-serif;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        font-family: Arial, Helvetica, sans-serif;
        letter-spacing: -0.01em;
    }
    .report-card {
        font-family: "Times New Roman", Times, serif;
        border: 1px solid #d6dae0;
        border-left: 4px solid #1a3a5c;
        border-radius: 3px;
        padding: 26px 30px 22px 30px;
        margin: 4px 0 18px 0;
        background: #ffffff;
    }
    .report-card .label {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.72rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #6b7280;
        margin: 0 0 8px 0;
    }
    .report-card .value {
        font-size: 2.05rem;
        line-height: 1.15;
        color: #111111;
        margin: 0 0 4px 0;
    }
    .report-card .confidence {
        font-size: 1.15rem;
        color: #1a3a5c;
        margin: 0 0 14px 0;
    }
    .report-card .band {
        display: inline-block;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.74rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid #1a3a5c;
        border-radius: 2px;
        padding: 3px 9px;
        color: #1a3a5c;
    }
    .report-card .foot {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.78rem;
        color: #6b7280;
        margin: 16px 0 0 0;
        padding-top: 12px;
        border-top: 1px solid #eceff2;
    }
    .section-title {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6b7280;
        margin: 6px 0 12px 0;
    }
    .helper {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.82rem;
        color: #6b7280;
        margin: -4px 0 14px 0;
    }
    .clock {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.78rem;
        color: #1a3a5c;
        text-align: right;
        padding: 0 2px 2px 0;
    }
    .model-note {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.8rem;
        line-height: 1.5;
        color: #4b5563;
    }
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1080px; }
    [data-testid="stSidebar"] { border-right: 1px solid #e6e9ec; }
</style>
"""


def format_nsm(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def confidence_band(probability):
    if probability >= 0.85:
        return "High confidence"
    if probability >= 0.60:
        return "Moderate confidence"
    return "Low confidence"


def build_dataset(frame, day_offset=1, weekstart_offset=1):
    days = {day: index + day_offset for index, day in enumerate(DAY_ORDER)}
    features = frame[NUMERIC_FEATURES].copy()
    features["Day_factorized"] = frame["Day_of_week"].map(days)
    features["WeekStatus_factorized"] = frame["WeekStatus"].map(
        {"Weekday": weekstart_offset, "Weekend": weekstart_offset + 1}
    )
    return features[FEATURE_COLUMNS]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as handle:
                model = pickle.load(handle)
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                return model, "model.pkl", None
        except Exception:
            problem = (
                "model.pkl could not be read and was ignored. Re-export it from "
                "Colab with pickle.dump and a round-trip check."
            )
        else:
            problem = None
    else:
        problem = None

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "No usable model was found. Place model.pkl in this folder, or keep "
            "Steel_industry_data.csv so the model can be trained."
        )

    frame = pd.read_csv(DATA_PATH)
    labels = frame["Load_Type"].map(
        {"Light_Load": 1, "Medium_Load": 2, "Maximum_Load": 3}
    ).astype(int)

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=18,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(build_dataset(frame), labels)

    target = LOCAL_FALLBACK
    if not os.access(BASE_DIR, os.W_OK):
        target = FALLBACK_PATH
    try:
        with open(target, "wb") as handle:
            pickle.dump(model, handle, protocol=4)
    except OSError:
        pass

    return model, "model_trained.pkl", problem


@st.cache_resource(show_spinner="Validating model on historical records...")
def validate_model(_model):
    if not os.path.exists(DATA_PATH):
        return None
    frame = pd.read_csv(DATA_PATH)
    labels = frame["Load_Type"].map(
        {"Light_Load": 1, "Medium_Load": 2, "Maximum_Load": 3}
    ).astype(int)
    predicted = _model.predict(build_dataset(frame))
    return float((predicted == labels).mean())


def sync_value(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]


def paired_input(label, minimum, maximum, step, default, key, number_format):
    slider_key = f"{key}_slider"
    box_key = f"{key}_box"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(default)
        st.session_state[box_key] = float(default)

    slider_col, box_col = st.columns([3, 1.25], vertical_alignment="bottom")

    with slider_col:
        st.slider(
            label,
            min_value=minimum,
            max_value=maximum,
            step=step,
            key=slider_key,
            on_change=sync_value,
            args=(slider_key, box_key),
        )

    with box_col:
        st.number_input(
            f"{label} value",
            min_value=minimum,
            max_value=maximum,
            step=step,
            key=box_key,
            on_change=sync_value,
            args=(box_key, slider_key),
            label_visibility="collapsed",
            format=number_format,
        )

    return float(st.session_state[slider_key])


def main():
    st.set_page_config(
        page_title="Steel Industry Load Classification",
        page_icon=None,
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        model, model_name, load_problem = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    st.title("Steel Industry Load Classification")
    st.markdown(
        '<p class="helper">Enter interval measurements to classify the load type '
        "as light, medium or maximum, with the model's confidence in the result.</p>",
        unsafe_allow_html=True,
    )

    if load_problem:
        st.warning(load_problem + " A model trained on the historical data is in use.")

    st.markdown('<div class="section-title">Input Parameters</div>', unsafe_allow_html=True)

    values = {}
    left, right = st.columns(2, gap="large")

    with left:
        for name, minimum, maximum, step, default, fmt in FEATURE_SPECS[:3]:
            key = name.replace("(", "").replace(")", "").replace(".", "_")
            values[name] = paired_input(name, minimum, maximum, step, default, key, fmt)

    with right:
        for name, minimum, maximum, step, default, fmt in FEATURE_SPECS[3:]:
            key = name.replace("(", "").replace(")", "").replace(".", "_")
            values[name] = paired_input(name, minimum, maximum, step, default, key, fmt)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    time_left, time_right = st.columns([3, 1.25], vertical_alignment="bottom")
    with time_left:
        values["NSM"] = paired_input("NSM (time of day)", 0, 86400, 900, 28800, "nsm", "%.0f")
    with time_right:
        st.markdown(
            f'<div class="clock">{format_nsm(values["NSM"])}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    day_col, week_col = st.columns(2, gap="large")
    with day_col:
        day = st.selectbox("Day of week", DAY_ORDER, index=0)
    with week_col:
        week = st.selectbox("Day type", ["Weekday", "Weekend"], index=0)

    values["Day_factorized"] = float(DAY_ORDER.index(day) + 1)
    values["WeekStatus_factorized"] = 1.0 if week == "Weekday" else 2.0

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    action_col, _ = st.columns([1, 4])
    with action_col:
        submitted = st.button("Predict Load Type", type="primary", use_container_width=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if submitted:
        vector = pd.DataFrame([[values[column] for column in FEATURE_COLUMNS]],
                              columns=FEATURE_COLUMNS)
        predicted = int(model.predict(vector)[0])
        probabilities = model.predict_proba(vector)[0]

        label = CLASS_LABELS.get(predicted, "Unknown")
        confidence = float(probabilities.max())

        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="report-card">
              <p class="label">Predicted Load Type</p>
              <p class="value">{label}</p>
              <p class="confidence">{confidence * 100:.1f}% confidence</p>
              <span class="band">{confidence_band(confidence)}</span>
              <p class="foot">Model output class {predicted} of 3.
              Confidence is the model's estimated probability for the class above.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="helper">Probability assigned to each load type.</p>',
            unsafe_allow_html=True,
        )
        breakdown = pd.DataFrame(
            {
                "Load Type": list(CLASS_LABELS.values()),
                "Probability": [float(probabilities[list(CLASS_LABELS).index(cls)])
                                for cls in sorted(CLASS_LABELS)],
            }
        ).set_index("Load Type")
        st.bar_chart(breakdown, height=230, color="#1a3a5c")

    accuracy = validate_model(model)

    with st.sidebar:
        st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="model-note">
              Algorithm<br>
              Random Forest Classifier<br><br>
              Estimators<br>
              {getattr(model, "n_estimators", 100)} trees<br><br>
              Features<br>
              {len(FEATURE_COLUMNS)} inputs
            </div>
            """,
            unsafe_allow_html=True,
        )

        if accuracy is not None:
            st.markdown('<div class="section-title">Validation</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="model-note">
                  Accuracy on historical records<br>
                  <strong>{accuracy * 100:.2f}%</strong><br>
                  Measured on all {len(pd.read_csv(DATA_PATH, usecols=["Load_Type"])):,}
                  rows in <em>Steel_industry_data.csv</em>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Source</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="model-note">Active model file<br><strong>{model_name}</strong></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'Times New Roman\', Times, serif; font-size:0.82rem;">',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                FEATURE_INFO, columns=["Feature", "Description"]
            ).set_index("Feature"),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        """
        <div style="margin-top:38px; padding-top:14px; border-top:1px solid #e6e9ec;
                   font-family:'Times New Roman', Times, serif; font-size:0.8rem;
                   color:#6b7280;">
          Class mapping: 1 = Light Load, 2 = Medium Load, 3 = Maximum Load.
          Confidence reflects the model's estimated probability, not a guarantee.
        </div>
        """,
        unsafe_allow_html=True,
    )


main()