import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="MedQA Dashboard", layout="wide")
st.title("📊 MedQA Data Dashboard")

# Paths
OFFLINE_DIR = "/mnt/object/data/dataset-split"
RETRAIN_DIR = "/mnt/object/data/production/retraining_data_transformed"

# Tabs
tab1, tab2 = st.tabs(["📂 Offline MedQuAD Data", "🔁 Retraining Data"])

# ========== TAB 1 ==========
with tab1:
    st.header("📂 Offline MedQuAD Data (Cleaned)")

    META_PATH = os.path.join(OFFLINE_DIR, "metadata.json")
    TRAIN_PATH = os.path.join(OFFLINE_DIR, "training", "training.json")

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            metadata = json.load(f)

        st.subheader("📝 Metadata Summary")
        st.json(metadata)

        # Drop stats
        dropped = metadata.get("dropped", {})
        if dropped:
            st.markdown("### ❌ Dropped Records Due to Missing/Blank Fields")
            drop_df = pd.DataFrame(list(dropped.items()), columns=["Type", "Count"])
            fig = px.bar(drop_df, x="Type", y="Count", title="Drop Statistics", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        # Split stats
        st.markdown("### 📦 Dataset Split Counts")
        split = metadata.get("split_counts", {})
        if split:
            st.write(split)
        else:
            st.warning("Split counts not found in metadata.")

        # Preview data
        st.markdown("### 🔍 Sample Questions from Training Set")
        df_train = pd.read_json(TRAIN_PATH, lines=True)

        # Filters
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

        # Question type histogram
        st.markdown("### 📘 Question Type Distribution")
        fig2 = px.histogram(df_train, x="question_type", title="Distribution of Question Types")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error("Metadata not found in dataset-split.")

# ========== TAB 2 ==========
with tab2:
    st.header("🔁 Retraining Data")

    available_versions = sorted(
        [v for v in os.listdir(RETRAIN_DIR) if v.startswith("v") and os.path.isdir(os.path.join(RETRAIN_DIR, v))],
        key=lambda x: int(x[1:])
    )

    if not available_versions:
        st.warning("No retraining versions found.")
    else:
        selected_version = st.selectbox("Select Version", available_versions)
        version_path = os.path.join(RETRAIN_DIR, selected_version)
        metadata_path = os.path.join(version_path, "metadata.json")
        data_path = os.path.join(version_path, "retraining_data.json")

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)

            st.subheader("📄 Metadata")
            st.json(meta)

            # Drop statistics
            dropped = meta.get("dropped", {})
            if dropped:
                st.markdown("### ❌ Dropped Records Breakdown")
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
            st.error("Cleaned retraining data not found for selected version.")
