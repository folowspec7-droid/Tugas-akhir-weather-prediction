from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Rain Tomorrow Predictor", page_icon="🌦️", layout="wide")
st.title("🌦️ Prediksi Hujan Besok — Australia")
st.caption("Demo klasifikasi biner menggunakan dataset Weather AUS. Bukan pengganti prakiraan resmi.")

DATA_PATH = Path(__file__).parent / "weatherAUS.csv"
MODEL_PATH = Path(__file__).parent / "rain_model.joblib"
TARGET = "RainTomorrow"
DROP_LEAKAGE = ["RISK_MM"]  # kebocoran target: terkait langsung dengan hujan esok hari

@st.cache_resource
def train_model(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")
    df = df.drop(columns=[c for c in DROP_LEAKAGE if c in df.columns])
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["DayOfYear_sin"] = np.sin(2*np.pi*df["Date"].dt.dayofyear/365.25)
        df["DayOfYear_cos"] = np.cos(2*np.pi*df["Date"].dt.dayofyear/365.25)
        df["Year"] = df["Date"].dt.year
        df = df.drop(columns=["Date"])
    y = df.pop(TARGET).astype(str)
    X = df
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("preprocess", prep), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    # For this demo app, stratified holdout is used to keep training responsive.
    # Use the provided notebook for chronological validation/test evaluation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    return model, list(X.columns), X_test, y_test, model.predict(X_test)

if not DATA_PATH.exists():
    st.error("Dataset belum ditemukan. Simpan file weatherAUS.csv di folder yang sama dengan app.py.")
    st.markdown("Sumber dataset: [Kaggle — Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)")
    st.stop()

try:
    model, features, X_test, y_test, y_pred = train_model(str(DATA_PATH))
except Exception as e:
    st.error(f"Gagal memuat/melatih model: {e}")
    st.stop()

with st.expander("Evaluasi demo (holdout acak; lihat notebook untuk evaluasi kronologis)", expanded=False):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test,y_pred):.3f}")
    c2.metric("Precision", f"{precision_score(y_test,y_pred,pos_label='Yes',zero_division=0):.3f}")
    c3.metric("Recall", f"{recall_score(y_test,y_pred,pos_label='Yes',zero_division=0):.3f}")
    c4.metric("F1-score", f"{f1_score(y_test,y_pred,pos_label='Yes',zero_division=0):.3f}")

st.subheader("Masukkan kondisi cuaca")
st.info("Pilih lokasi dan isi nilai pengamatan. Fitur yang tidak ditampilkan akan diisi dengan nilai median/modus dari pipeline.")
col1, col2, col3 = st.columns(3)
values = {}
defaults = {
    "Location": ("categorical", "Sydney"),
    "MinTemp": ("number", 15.0), "MaxTemp": ("number", 24.0),
    "Rainfall": ("number", 0.0), "Evaporation": ("number", 4.0),
    "Sunshine": ("number", 7.0), "WindGustDir": ("categorical", "N"),
    "WindGustSpeed": ("number", 35.0), "WindDir9am": ("categorical", "N"),
    "WindDir3pm": ("categorical", "N"), "WindSpeed9am": ("number", 15.0),
    "WindSpeed3pm": ("number", 20.0), "Humidity9am": ("number", 65.0),
    "Humidity3pm": ("number", 50.0), "Pressure9am": ("number", 1015.0),
    "Pressure3pm": ("number", 1012.0), "Cloud9am": ("number", 4.0),
    "Cloud3pm": ("number", 4.0), "Temp9am": ("number", 18.0),
    "Temp3pm": ("number", 23.0), "RainToday": ("categorical", "No"),
    "DayOfYear_sin": ("number", 0.0), "DayOfYear_cos": ("number", 1.0),
    "Year": ("number", 2017)
}
for i, feat in enumerate(features):
    col = [col1,col2,col3][i%3]
    kind, default = defaults.get(feat, ("categorical", "Unknown"))
    if feat in ("Location","WindGustDir","WindDir9am","WindDir3pm","RainToday"):
        if feat == "Location":
            opts = sorted(pd.read_csv(DATA_PATH, usecols=["Location"])["Location"].dropna().unique().tolist())
        elif feat.startswith("Wind"):
            opts = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
        else:
            opts = ["No","Yes"]
        val = default if default in opts else opts[0]
        values[feat] = col.selectbox(feat, opts, index=opts.index(val), key=feat)
    elif kind == "number":
        values[feat] = col.number_input(feat, value=float(default), key=feat)
    else:
        values[feat] = col.text_input(feat, value=str(default), key=feat)

if st.button("Prediksi hujan besok", type="primary"):
    row = pd.DataFrame([{f: values.get(f, np.nan) for f in features}])
    pred = model.predict(row)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if "Yes" in classes:
            prob = float(model.predict_proba(row)[0][classes.index("Yes")])
    if pred == "Yes":
        st.warning("🌧️ Model memprediksi: kemungkinan hujan besok.")
    else:
        st.success("☀️ Model memprediksi: kemungkinan tidak hujan besok.")
    if prob is not None:
        st.metric("Probabilitas hujan (model)", f"{prob:.1%}")
    st.caption("Hasil adalah estimasi model dari data historis, bukan jaminan cuaca aktual.")
