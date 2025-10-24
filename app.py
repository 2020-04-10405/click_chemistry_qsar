# app.py
# Streamlit QSAR App — 512-bit Morgan fingerprints (radius=2) with AD check
# No molecule images, no AD plot

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os
from datetime import datetime

# ----------------------
# Streamlit configuration
# ----------------------
st.set_page_config(
    page_title="Alpha-Glucosidase QSAR Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------
# Config / paths
# ----------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "GradientBoosting_Tuned_Model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "StandardScaler.joblib")
CENTROID_PATH = os.path.join(MODELS_DIR, "AD_centroid.npy")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "AD_threshold.npy")

FP_BITS = 512
FP_RADIUS = 2

# ----------------------
# Cached loader
# ----------------------
@st.cache_resource
def load_assets():
    missing = [p for p in (MODEL_PATH, SCALER_PATH, CENTROID_PATH, THRESHOLD_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model/AD files: {missing}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    centroid = np.load(CENTROID_PATH)
    threshold = float(np.load(THRESHOLD_PATH))
    return {"model": model, "scaler": scaler, "centroid": centroid, "threshold": threshold}

# ----------------------
# Fingerprint + predict
# ----------------------
def smiles_to_fp_array(smiles: str, n_bits: int = FP_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def predict_single(smiles: str, assets: dict):
    X = smiles_to_fp_array(smiles)
    if X is None:
        return {"error": "Invalid SMILES"}
    scaler = assets["scaler"]
    model = assets["model"]
    centroid = assets["centroid"]
    threshold = assets["threshold"]

    X_scaled = scaler.transform([X])
    distance = float(np.linalg.norm(X_scaled - centroid))
    in_ad = distance <= threshold
    y_pred = float(model.predict(X_scaled)[0])
    return {
        "smiles": smiles,
        "predicted_pIC50": y_pred,
        "distance": distance,
        "in_ad": bool(in_ad),
        "ad_status": "Within AD" if in_ad else "Outside AD",
        "threshold": threshold
    }

def predict_batch(smiles_list, assets: dict):
    results = []
    for smi in smiles_list:
        res = predict_single(smi, assets)
        if "error" in res:
            results.append({
                "smiles": smi,
                "predicted_pIC50": np.nan,
                "distance": np.nan,
                "ad_status": res["error"],
                "in_ad": False
            })
        else:
            results.append({
                "smiles": res["smiles"],
                "predicted_pIC50": res["predicted_pIC50"],
                "distance": res["distance"],
                "ad_status": res["ad_status"],
                "in_ad": res["in_ad"]
            })
    return pd.DataFrame(results)

# ----------------------
# Helper: download link
# ----------------------
def get_csv_download_link(df: pd.DataFrame, name="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{name}">📥 Download predictions</a>'

def ad_confidence_percent(distance, threshold):
    if distance == 0 or distance is None or np.isnan(distance):
        return 100
    pct = min((threshold / distance) * 100, 100)
    return int(round(pct))

# ----------------------
# UI
# ----------------------
def app_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        logo = os.path.join("static", "logo_ud.png")
        if os.path.exists(logo):
            st.image(logo, width=90, use_container_width=False)
    with col2:
        st.markdown("<h1 style='color:#004080'>Alpha-Glucosidase QSAR Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<div style='color:#333'>Predict pIC50 using a GradientBoosting model trained on 512-bit Morgan fingerprints (radius=2). Applicability Domain (AD) evaluated by Euclidean distance.</div>", unsafe_allow_html=True)

def sidebar_inputs():
    st.sidebar.header("Input options")
    single_smiles = st.sidebar.text_input("Single SMILES input", value="", placeholder="C1=CC=CC=C1")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (column 'smiles' or headerless)", type=["csv","txt"])
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Predict")
    return single_smiles, uploaded_file, run_button

# ----------------------
# Main logic
# ----------------------
def main():
    try:
        assets = load_assets()
    except FileNotFoundError:
        st.error("Missing required model files in the `models/` directory.")
        st.stop()

    app_header()
    st.markdown("---")

    single_smiles, uploaded_file, run_button = sidebar_inputs()

    st.subheader("Input")
    st.markdown("Provide a single SMILES in the sidebar or upload a CSV file containing SMILES (one per row).")
    st.markdown("Example: `CCO`, `CC(=O)Oc1ccccc1C(=O)O`")

    st.subheader("Model info")
    st.write(f"Descriptor: **{FP_BITS}-bit Morgan fingerprint (radius={FP_RADIUS})**")
    st.write(f"AD threshold: **{assets['threshold']:.3f}**")
    st.write(f"Model: **{os.path.basename(MODEL_PATH)}**")

    # Prepare SMILES
    smiles_list = []
    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            cols = [c for c in df_in.columns if "smiles" in c.lower()]
            smiles_list = df_in[cols[0]].astype(str).tolist() if cols else df_in.iloc[:,0].astype(str).tolist()
        except Exception:
            uploaded_file.seek(0)
            txt = uploaded_file.read().decode("utf-8")
            smiles_list = [l.strip() for l in txt.splitlines() if l.strip()]
    elif single_smiles:
        smiles_list = [single_smiles.strip()]

    if run_button:
        if not smiles_list:
            st.warning("No SMILES provided. Please enter a SMILES or upload a CSV.")
        else:
            with st.spinner("Predicting..."):
                df_preds = predict_batch(smiles_list, assets)

            st.subheader("Predictions")
            st.dataframe(df_preds.style.format({"predicted_pIC50": "{:.3f}", "distance": "{:.3f}"}), use_container_width=True)

            # Download link
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.markdown(get_csv_download_link(df_preds, name=f"predictions_{ts}.csv"), unsafe_allow_html=True)

            # If single prediction, show details
            if len(df_preds) == 1:
                r = df_preds.iloc[0]
                st.markdown("### Result detail")
                st.write(f"SMILES: `{r['smiles']}`")
                st.write(f"Predicted pIC50: **{r['predicted_pIC50']:.3f}**")
                st.write(f"AD status: **{r['ad_status']}** (distance={r['distance']:.3f})")
                pct = ad_confidence_percent(r['distance'], assets['threshold'])
                st.progress(min(100, pct))
                st.write(f"{pct}% AD confidence")

    else:
        st.info("Enter a SMILES or upload a CSV, then click **Predict** in the sidebar.")

    # ----------------------
    # Footer branding
    # ----------------------
    st.markdown("---")
    st.caption("Developed by Mr. Nyunyu Junior | Department of Chemistry | Mkwawa University College of Education | University of Dar es Salaam | 2025")
    st.caption("Descriptor: 512-bit Morgan fingerprint (radius=2)")

if __name__ == "__main__":
    main()
