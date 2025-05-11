import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="MedQA Dashboard", layout="wide")
st.title("📊 MedQA Data Dashboard")

OFFLINE_DIR = "/mnt/object/data/dataset-split"
RETRAIN_DIR = "/mnt/object/data/production/retraining_data_transformed"

tab1, tab2 = st.tabs(["📂 Offline MedQuAD Data", "🔁 Retraining Data"])

# --- TAB 1: OFFLINE DATA ---
with tab1:
    st.header("📂 Offline MedQuAD Data (Cleaned)")
    meta_path = os.path.join(OFFLINE_DIR, "metadata.json")
    train_path = os.path.join(OFFLINE_DIR, "training", "training.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        st.subheader("📝 Metadata Summary")
        st.json(metadata)

        dropped = metadata.get("dropped", {})
        if dropped:
            st.markdown("### ❌ Dropped Records")
            drop_df = pd.DataFrame(list(dropped.items()), columns=["Type", "Count"])
            fig = px.bar(drop_df, x="Type", y="Count", title="Drop Statistics", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📦 Dataset Split Counts")
        split = metadata.get("split_counts", {})
        st.write(split)

        df_train = pd.read_json(train_path, lines=True)
        st.markdown("### 🔍 Training Set Explorer")
        qtype_options = df_train["question_type"].unique()
        selected_types = st.multiselect("Filter by question_type", qtype_options)
        keyword = st.text_input("Keyword in question")

        filtered_df = df_train.copy()
        if selected_types:
            filtered_df = filtered_df[filtered_df["question_type"].isin(selected_types)]
        if keyword:
            filtered_df = filtered_df[filtered_df["question"].str.contains(keyword, case=False, na=False)]

        st.write(f"Showing {len(filtered_df)} / {len(df_train)} records")
        st.dataframe(filtered_df.sample(min(10, len(filtered_df))), use_container_width=True)

        st.markdown("### 📘 Question Type Distribution")
        fig2 = px.histogram(df_train, x="question_type", title="Distribution of Question Types")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error("metadata.json not found at dataset-split/")

# --- TAB 2: RETRAINING DATA ---
with tab2:
    st.header("🔁 Retraining Data")
    versions = sorted(
        [v for v in os.listdir(RETRAIN_DIR) if v.startswith("v") and os.path.isdir(os.path.join(RETRAIN_DIR, v))],
        key=lambda x: int(x[1:])
    )

    if not versions:
        st.warning("No retraining versions found.")
    else:
        selected_version = st.selectbox("Select Version", versions)
        version_path = os.path.join(RETRAIN_DIR, selected_version)
        meta_path = os.path.join(version_path, "metadata.json")
        data_path = os.path.join(version_path, "retraining_data.json")

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            st.subheader("📄 Metadata")
            st.json(meta)

            dropped = meta.get("dropped", {})
            if dropped:
                st.markdown("### ❌ Dropped Records")
                dropped_df = pd.DataFrame(list(dropped.items()), columns=["Type", "Count"])
                fig = px.bar(dropped_df, x="Type", y="Count", title="Drop Statistics", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Metadata not found for selected version.")

        if os.path.exists(data_path):
            df = pd.read_json(data_path, lines=True)

            st.markdown("### 📘 Question Type Distribution")
            if "question_type" in df.columns:
                fig2 = px.histogram(df, x="question_type", title="Question Type Histogram")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 🔍 Record Explorer")
            qtypes = df["question_type"].dropna().unique().tolist() if "question_type" in df.columns else []
            selected_qtypes = st.multiselect("Filter by question_type", qtypes)
            keyword = st.text_input("Keyword in question")

            filtered_df = df.copy()
            if selected_qtypes:
                filtered_df = filtered_df[filtered_df["question_type"].isin(selected_qtypes)]
            if keyword:
                filtered_df = filtered_df[filtered_df["question"].str.contains(keyword, case=False, na=False)]

            st.write(f"Showing {len(filtered_df)} / {len(df)} records")
            st.dataframe(filtered_df.sample(min(10, len(filtered_df))), use_container_width=True)
        else:
            st.error("retraining_data.json not found for selected version.")
