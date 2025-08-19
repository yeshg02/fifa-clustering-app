import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# -------------------------------
# Load Pre-trained Model & Artifacts
# -------------------------------
dbscan_model = joblib.load("dbscan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # list of input features used in training

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="⚽ FIFA Player Clustering", layout="wide")
st.title("⚽ FIFA Player Clustering App")
st.write("Explore and group football players using the **DBSCAN clustering model** based on their stats.")

st.sidebar.header("📘 About")
st.sidebar.markdown(
    """
    Upload a dataset **or** manually enter player stats.  
    The app will cluster the player(s) and assign football-style roles  
    based on skill-based features from FIFA 20.
    """
)

# -------------------------------
# Helper Function: Assign Role Labels
# -------------------------------
def label_cluster(stats_row):
    """Assigns a football-style role based on key skill stats."""
    attack_feats = ["attacking_finishing", "power_shot_power", "skill_dribbling"]
    defense_feats = ["defending", "power_strength"]
    midfield_feats = ["passing", "mentality_vision"]

    def avg_feats(row, feats):
        return np.mean([row[f] for f in feats if f in row])

    attack = avg_feats(stats_row, attack_feats)
    defense = avg_feats(stats_row, defense_feats)
    midfield = avg_feats(stats_row, midfield_feats)

    if attack > defense and attack > midfield:
        return "⚽ Attacker"
    elif defense > attack and defense > midfield:
        return "🛡️ Defender"
    elif midfield > attack and midfield > defense:
        return "🎯 Playmaker"
    else:
        return "🔄 All-Rounder"

# -------------------------------
# Input Mode Selection
# -------------------------------
st.subheader("🎮 Choose Input Mode")
mode = st.radio("Select Input Method", ["📂 Upload CSV", "📝 Manual Entry"])

# -------------------------------
# 📂 Upload CSV Mode
# -------------------------------
if mode == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with player stats", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        # Align columns with training
        df_input = df.reindex(columns=feature_names, fill_value=0)
        X_scaled = scaler.transform(df_input)

        # Predict clusters
        cluster_labels = dbscan_model.fit_predict(X_scaled)
        df["Cluster"] = cluster_labels

        # Add 2D PCA projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df["PCA1"], df["PCA2"] = X_pca[:, 0], X_pca[:, 1]

        st.write("### 📊 Cluster Distribution")
        st.dataframe(df["Cluster"].value_counts().reset_index(names=["Cluster", "Count"]))

        # Plot clusters
        st.write("### 🎨 PCA Cluster Visualization")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set1", alpha=0.8)
        plt.title("Clusters (2D PCA Projection)")
        st.pyplot(plt)

        # Download button
        st.write("### 💾 Download Results")
        st.download_button(
            label="Download Clustered CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="clustered_players.csv",
            mime="text/csv"
        )

# -------------------------------
# 📝 Manual Entry Mode
# -------------------------------
elif mode == "📝 Manual Entry":
    st.subheader("Enter Player Stats Manually")

    player_input = {}
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            player_input[feature] = st.slider(feature.replace("_", " ").title(), 0, 100, 50)

    # Prediction on button click
    if st.button("🔮 Predict Player Role"):
        player_df = pd.DataFrame([player_input])
        player_scaled = scaler.transform(player_df)
        cluster_label = dbscan_model.fit_predict(player_scaled)[0]
        role = label_cluster(player_input)

        st.write("### 📋 Prediction Result")
        st.write(f"**Cluster:** {cluster_label}")
        st.write(f"**Predicted Role:** {role}")
        st.dataframe(player_df.T.rename(columns={0: "Value"}))
