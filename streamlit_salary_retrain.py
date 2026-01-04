import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS = ["YearsExperience", "Salary"]

def clean_labeled_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + clean incoming labeled dataset."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}. Required: {REQUIRED_COLS}")

    out = df[REQUIRED_COLS].copy()
    out["YearsExperience"] = pd.to_numeric(out["YearsExperience"], errors="coerce")
    out["Salary"] = pd.to_numeric(out["Salary"], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLS)

    if len(out) < 2:
        raise ValueError("Not enough valid rows after cleaning. Provide at least 2 valid rows.")
    return out

def build_model() -> Pipeline:
    """Regression pipeline."""
    # scaler is optional for single feature, but safe for future multi-feature extension
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

def fit_model(train_df: pd.DataFrame) -> Pipeline:
    X = train_df[["YearsExperience"]].values
    y = train_df["Salary"].values
    model = build_model()
    model.fit(X, y)
    return model

def eval_model(model: Pipeline, eval_df: pd.DataFrame) -> dict:
    X = eval_df[["YearsExperience"]].values
    y_true = eval_df["Salary"].values
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        # User asked for "accuracy" -> for regression, R² is a common accuracy-like score
        "R2 (Accuracy-like)": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "Rows": int(len(eval_df))
    }

def predict_salary(model: Pipeline, years_exp: float) -> float:
    X = np.array([[years_exp]], dtype=float)
    return float(model.predict(X)[0])


# -----------------------------
# Session State Init
# -----------------------------
if "train_data" not in st.session_state:
    st.session_state.train_data = None  # base + accumulated data

if "models" not in st.session_state:
    # list of dicts: {name, model, trained_rows, metrics_on_latest_eval(optional)}
    st.session_state.models = []

if "active_model_name" not in st.session_state:
    st.session_state.active_model_name = None

if "latest_eval_df" not in st.session_state:
    st.session_state.latest_eval_df = None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Salary Prediction (Old vs New Model)", layout="wide")
st.title("Salary Prediction (Continuous Retraining: Old vs New Model)")

st.markdown(
    """
**Workflow**
1) Upload **Base Training CSV** → trains **Old Model**  
2) Upload **New Labeled CSV** (with Salary) → evaluates Old Model, retrains → creates **New Model**  
3) Choose **Old/New model** for predictions  
4) Repeat step #2 anytime (recurrent cycle)
"""
)

colA, colB = st.columns(2)

# -----------------------------
# 1) Base Training Upload
# -----------------------------
with colA:
    st.subheader("1) Upload Base Training CSV")
    base_file = st.file_uploader("Base training file (YearsExperience, Salary)", type=["csv"], key="base_upload")

    if base_file is not None:
        try:
            base_df = pd.read_csv(base_file)
            base_df = clean_labeled_df(base_df)

            st.session_state.train_data = base_df.copy()

            # Train Old Model
            old_model = fit_model(st.session_state.train_data)
            st.session_state.models = [{
                "name": "Old Model (Base)",
                "model": old_model,
                "trained_rows": len(st.session_state.train_data),
                "metrics_on_latest_eval": None
            }]
            st.session_state.active_model_name = "Old Model (Base)"

            st.success(f"Old Model trained on {len(st.session_state.train_data)} rows.")
            st.dataframe(base_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"Base training upload failed: {e}")

# -----------------------------
# 2) New Labeled Data Upload (Evaluate + Retrain)
# -----------------------------
with colB:
    st.subheader("2) Upload New Labeled Data (Evaluation + Retraining)")
    new_file = st.file_uploader("New labeled file (YearsExperience, Salary)", type=["csv"], key="new_upload")

    if new_file is not None:
        if st.session_state.train_data is None or len(st.session_state.models) == 0:
            st.warning("Please upload Base Training CSV first (to create the Old Model).")
        else:
            try:
                new_df = pd.read_csv(new_file)
                new_df = clean_labeled_df(new_df)
                st.session_state.latest_eval_df = new_df.copy()

                # Evaluate existing models on new_df
                for m in st.session_state.models:
                    m["metrics_on_latest_eval"] = eval_model(m["model"], new_df)

                # Retrain a brand new model using (existing accumulated train_data + new_df)
                combined = pd.concat([st.session_state.train_data, new_df], ignore_index=True).drop_duplicates()
                st.session_state.train_data = combined

                new_model = fit_model(st.session_state.train_data)

                # Name it with version count
                version = sum(1 for m in st.session_state.models if m["name"].startswith("New Model v")) + 1
                new_name = f"New Model v{version} (Base + Updates)"

                # Evaluate the newly trained model on the latest eval set too
                new_metrics = eval_model(new_model, new_df)

                st.session_state.models.append({
                    "name": new_name,
                    "model": new_model,
                    "trained_rows": len(st.session_state.train_data),
                    "metrics_on_latest_eval": new_metrics
                })

                st.success(
                    f"{new_name} trained on total {len(st.session_state.train_data)} rows. "
                    f"Evaluated on new upload: {len(new_df)} rows."
                )
                st.dataframe(new_df.head(20), use_container_width=True)

            except Exception as e:
                st.error(f"New labeled upload failed: {e}")


# -----------------------------
# 4) Model Comparison + Selection
# -----------------------------
st.subheader("4) Compare Models (Old vs New) + Select Active Model")

if len(st.session_state.models) == 0:
    st.info("Upload Base Training CSV to create the Old Model.")
else:
    # Build comparison table
    rows = []
    for m in st.session_state.models:
        metrics = m.get("metrics_on_latest_eval") or {}
        rows.append({
            "Model": m["name"],
            "Trained Rows": m["trained_rows"],
            "R2 (Accuracy-like)": metrics.get("R2 (Accuracy-like)", None),
            "MAE": metrics.get("MAE", None),
            "RMSE": metrics.get("RMSE", None),
            "Eval Rows": metrics.get("Rows", None),
        })

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True)

    model_names = [m["name"] for m in st.session_state.models]
    # Keep current selection if exists
    if st.session_state.active_model_name not in model_names:
        st.session_state.active_model_name = model_names[0]

    chosen = st.radio(
        "Choose which model to use for the next prediction:",
        model_names,
        index=model_names.index(st.session_state.active_model_name),
        horizontal=True
    )
    st.session_state.active_model_name = chosen

# -----------------------------
# 5) Prediction using selected model
# -----------------------------
st.subheader("5) Predict Salary (uses selected model)")

if len(st.session_state.models) == 0:
    st.info("Upload Base Training CSV first.")
else:
    active = next((m for m in st.session_state.models if m["name"] == st.session_state.active_model_name), None)
    if active is None:
        st.error("Active model not found. Please select again.")
    else:
        exp = st.slider("Years of Experience", 0.0, 40.0, 3.0, 0.5)
        try:
            pred = predict_salary(active["model"], exp)
            st.success(f"**{active['name']}** predicts salary for **{exp:.1f} yrs**: **₹{pred:,.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------------
# Debug / Reset
# -----------------------------
with st.expander("Utilities"):
    st.write("Training rows stored in memory:", None if st.session_state.train_data is None else len(st.session_state.train_data))
    if st.button("Reset everything"):
        st.session_state.train_data = None
        st.session_state.models = []
        st.session_state.active_model_name = None
        st.session_state.latest_eval_df = None
        st.rerun()
