# app.py
import os
import pickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Try to use LightGBM if available
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Academix")

# ---------------------------
# Branding / Title
# ---------------------------
st.title("🎓 Academix – Where Data Meets Education")
st.markdown("An interactive dashboard for student performance insights and ML-based score prediction.")

DATA_PATH = "Student_scores.csv"
MODEL_PATH = "best_model.pkl"


# Utility functions
# ---------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    return df

def train_and_save_model(df, features, targets, model_path=MODEL_PATH, random_state=42):
    # your existing training logic
    pass  

# Utility to safely sort unique values even if mixed types (str, float, NaN)
def safe_sorted(unique_vals):
    return sorted([str(v) for v in unique_vals if pd.notnull(v)])
def train_and_save_model(df, features, targets, model_path=MODEL_PATH, random_state=42):
    

    # Prepare X, y
    X = pd.get_dummies(df[features], drop_first=False)
    y = df[targets].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    if LGB_AVAILABLE:
        base = lgb.LGBMRegressor(random_state=random_state)
    else:
        base = GradientBoostingRegressor(random_state=random_state)

    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    # Save both model and columns for future use
    saved = {"model": model, "columns": X.columns.tolist(), "features": features}
    joblib.dump(saved, model_path)

    return model, X_test, y_test, y_pred, rmse, r2, X.columns.tolist()

def load_model_if_exists(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        try:
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                return loaded["model"], loaded.get("columns", None), loaded.get("features", None)
            else:
                return loaded, None, None
        except Exception as e:
            st.warning(f"Could not load model ({e}). Will retrain if needed.")
            return None, None, None
    return None, None, None

def build_input_vector(user_inputs, columns, features):
    vec = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    for feat, val in user_inputs.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            if feat in columns:
                vec.at[0, feat] = val
        else:
            key = f"{feat}_{val}"
            if key in columns:
                vec.at[0, key] = 1
            else:
                matches = [c for c in columns if c.startswith(f"{feat}_")]
                for m in matches:
                    if str(val) in m:
                        vec.at[0, m] = 1
                        break
    return vec

# ---------------------------
# Load dataset
# ---------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. Please place `Student_scores.csv` in the app folder.")
    st.stop()

df = load_data(DATA_PATH)

# Detect available target columns & features
default_targets = [c for c in ["MathScore", "ReadingScore", "WritingScore"] if c in df.columns]
if not default_targets:
    st.error("No target columns (MathScore/ReadingScore/WritingScore) found in the dataset.")
    st.stop()

candidate_features = ["Gender", "ParentEduc", "WklyStudyHours", "ParentMaritalStatus", "EthnicGroup"]
features = [f for f in candidate_features if f in df.columns]

# ---------------------------
# Sidebar Filters (safe version)
# ---------------------------
st.sidebar.header("Filters & Training Options")

def safe_multiselect(label, series):
    if series is None:
        return []
    options = series.dropna().astype(str).unique().tolist()
    return st.sidebar.multiselect(label, options=sorted(options), default=sorted(options))

sel_gender = safe_multiselect("Gender", df["Gender"] if "Gender" in df.columns else None)
sel_ethnic = safe_multiselect("Ethnic Group", df["EthnicGroup"] if "EthnicGroup" in df.columns else None)
sel_parent_educ = safe_multiselect("Parent Education", df["ParentEduc"] if "ParentEduc" in df.columns else None)

df_filtered = df.copy()
if sel_gender:
    df_filtered = df_filtered[df_filtered["Gender"].astype(str).isin(sel_gender)]
if sel_ethnic:
    df_filtered = df_filtered[df_filtered["EthnicGroup"].astype(str).isin(sel_ethnic)]
if sel_parent_educ:
    df_filtered = df_filtered[df_filtered["ParentEduc"].astype(str).isin(sel_parent_educ)]

# ---------------------------
# Overview & KPIs
# ---------------------------
st.subheader("Dataset overview")
left, right = st.columns([3,1])
with left:
    st.dataframe(df_filtered.head(200))
with right:
    st.metric("Rows", len(df_filtered))
    for t in default_targets:
        st.metric(f"Avg {t}", f"{df_filtered[t].mean():.2f}")

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Visualizations")
col1, col2 = st.columns(2)
with col1:
    if "Gender" in df_filtered.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtered, x="Gender", ax=ax)
        ax.bar_label(ax.containers[0])
        st.pyplot(fig)
with col2:
    if "EthnicGroup" in df_filtered.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtered, x="EthnicGroup", ax=ax)
        ax.bar_label(ax.containers[0])
        st.pyplot(fig)

if "ParentEduc" in df_filtered.columns:
    st.subheader("Parent Education vs Average Scores")
    gb = df_filtered.groupby("ParentEduc")[default_targets].mean()
    fig, ax = plt.subplots(figsize=(6,2 + 0.6*len(gb)))
    sns.heatmap(gb, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

if "WklyStudyHours" in df_filtered.columns:
    st.subheader("Boxplot: Scores by Weekly Study Hours")
    try:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_filtered, x="WklyStudyHours", y=default_targets[0], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception:
        pass

# ---------------------------
# Model Handling
# ---------------------------
st.subheader("Model status")

model, model_cols, model_features = load_model_if_exists()
model_loaded = model is not None

if model_loaded:
    st.success("Loaded existing model from disk.")
    if model_cols is None:
        st.info("Model columns not saved with model. Will infer from current data.")
else:
    st.info("No saved model found. Click 'Train model' to train on the dataset.")

if st.button("Train model on full dataset now"):
    if not features:
        st.error("No features found to train on.")
    else:
        with st.spinner("Training model..."):
            trained_model, X_test, y_test, y_pred, rmse, r2, saved_columns = train_and_save_model(df, features, default_targets)
        st.success(f"Trained and saved model. RMSE={rmse:.2f}, R²={r2:.2f}")
        model, model_cols, model_features = trained_model, saved_columns, features
        model_loaded = True

if st.button("Retrain model on CURRENT FILTERED dataset"):
    if not features:
        st.error("No features found to train on.")
    else:
        if len(df_filtered) < 10:
            st.warning("Filtered dataset has less than 10 rows; results may be poor.")
        with st.spinner("Training on filtered data..."):
            trained_model, X_test, y_test, y_pred, rmse, r2, saved_columns = train_and_save_model(df_filtered, features, default_targets)
        st.success(f"Retrained and saved model. RMSE={rmse:.2f}, R²={r2:.2f}")
        model, model_cols, model_features = trained_model, saved_columns, features
        model_loaded = True

# ---------------------------
# Prediction panel
# ---------------------------
st.subheader("Predict Student Scores (Math, Reading, Writing)")

input_cols = {}
form = st.form(key="predict_form")
for feat in features:
    unique_vals = df[feat].dropna().unique().tolist()
    if pd.api.types.is_numeric_dtype(df[feat]) or all(isinstance(x, (int, float, np.integer, np.floating)) for x in unique_vals):
        default_val = float(np.nanmedian(df[feat].dropna())) if len(df[feat].dropna())>0 else 0.0
        input_cols[feat] = form.number_input(feat, value=float(default_val))
    else:
        input_cols[feat] = form.selectbox(feat, options=sorted(map(str, unique_vals)))

submitted = form.form_submit_button("Predict")

if submitted:
    if not model_loaded:
        st.error("No model available. Please train the model first.")
    else:
        cols = model_cols if model_cols is not None else pd.get_dummies(df[features], drop_first=False).columns.tolist()
        input_vector = build_input_vector(input_cols, cols, features)
        input_vector = input_vector[cols]
        pred = model.predict(input_vector)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        st.success("Predictions:")
        for i, t in enumerate(default_targets):
            st.metric(t, f"{pred[0, i]:.2f}")

# ---------------------------
# Predicted vs Actual
# ---------------------------
if model_loaded and (model_cols is not None):
    st.subheader("Predicted vs Actual (holdout sample if available)")
    try:
        X_all = pd.get_dummies(df[features], drop_first=False)
        for c in model_cols:
            if c not in X_all.columns:
                X_all[c] = 0
        X_all = X_all[model_cols]
        y_all = df[default_targets]
        y_pred_all = model.predict(X_all)
        rmses = {t: np.sqrt(mean_squared_error(y_all[t], y_pred_all[:, i])) for i, t in enumerate(default_targets)}
        st.write("RMSE per target (on full dataset):")
        st.write(rmses)

        fig, axes = plt.subplots(1, len(default_targets), figsize=(5*len(default_targets), 4))
        if len(default_targets) == 1:
            axes = [axes]
        for i, t in enumerate(default_targets):
            axes[i].scatter(y_all[t], y_pred_all[:, i], alpha=0.4)
            axes[i].set_xlabel("Actual " + t)
            axes[i].set_ylabel("Predicted " + t)
            axes[i].set_title(f"{t}: Actual vs Predicted")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not produce Pred vs Actual: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
#st.markdown("**Notes:**\n- If you already have a saved model, the app will attempt to load it. If not, train with the button.  \n- Training will save `best_model.pkl` in this folder.  \n- If your dataset uses different column names, update `candidate_features` at the top of this file.")

# Motivational quotes
st.markdown("💡 **For Students:** *Education is the most powerful weapon which you can use to change the world.* – Nelson Mandela")
st.markdown("📚 **For Teachers:** *A good teacher can inspire hope, ignite the imagination, and instill a love of learning.* – Brad Henry")
st.markdown("👨‍👩‍👧 **For Parents:** *Behind every young child who believes in themselves is a parent who believed first.* – Matthew Jacobson")
