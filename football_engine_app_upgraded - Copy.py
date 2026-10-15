import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# FOOTBALL ARCHETYPE LABELS
# ============================================================

ARCHETYPE_NAMES = {
    "Forwards": {
        0: "Creative / Progressive Forward",
        1: "Finishing Forward"
    },

    "Midfielders": {
        0: "Deep Playmaker",
        1: "Box-to-Box Midfielder"
    },

    "Defenders": {
        0: "Ball-Playing Defender",
        1: "Defensive Defender"
    },

    "Multi-position": {
        0: "Two-Way Utility Player",
        1: "Possession-Oriented Utility Player"
    }
}


def get_archetype_name(position, cluster):
    try:
        return ARCHETYPE_NAMES[position][int(cluster)]
    except (KeyError, TypeError, ValueError):
        return "Unclassified"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Football Intelligence Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD ENGINE
# ============================================================

@st.cache_resource
def load_engine():
    return joblib.load("football_intelligence_engine.pkl")


try:
    engine = load_engine()
except Exception as e:
    st.error("The model bundle could not be loaded.")
    st.code(str(e))
    st.stop()

dashboard_data = engine["dashboard_data"]
similarity_data = engine["similarity_data"]
cluster_results = engine["cluster_results"]
priority_features = engine["priority_features"]

# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        return default if not np.isfinite(value) else value
    except (TypeError, ValueError):
        return default


def get_similarity_table(position, player_name):
    info = similarity_data[position]
    players = pd.Series(info["players"]).astype(str)
    matrix = np.asarray(info["matrix"], dtype=float)

    matches = players.str.casefold() == str(player_name).casefold()
    if not matches.any():
        return pd.DataFrame(columns=["Player", "Similarity"])

    idx = int(np.flatnonzero(matches.to_numpy())[0])
    scores = matrix[idx]

    result = pd.DataFrame({
        "Player": players,
        "Similarity": scores
    })

    result = result[
        result["Player"].str.casefold() != str(player_name).casefold()
    ].copy()

    result["Similarity"] = pd.to_numeric(
        result["Similarity"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return result.sort_values(
        "Similarity", ascending=False
    ).reset_index(drop=True)


def get_similarity_score(position, player_name):
    info = similarity_data[position]
    players = pd.Series(info["players"]).astype(str)
    matrix = np.asarray(info["matrix"], dtype=float)

    matches = players.str.casefold() == str(player_name).casefold()
    if not matches.any():
        return pd.Series(dtype=float)

    idx = int(np.flatnonzero(matches.to_numpy())[0])
    scores = pd.Series(matrix[idx], index=players)

    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0)
    return scores


def percentile_profile(df, features, player_row):
    values = []
    labels = []

    for feature in features:
        if feature not in df.columns:
            continue

        series = pd.to_numeric(df[feature], errors="coerce")
        value = safe_float(player_row.get(feature, 0))

        valid = series.dropna()
        if len(valid) <= 1:
            percentile = 50.0
        else:
            percentile = float((valid <= value).mean() * 100)

        values.append(percentile)
        labels.append(feature)

    return pd.DataFrame({
        "Feature": labels,
        "Percentile": values
    })


def make_radar(profile_df, title):
    if profile_df.empty:
        return None

    categories = profile_df["Feature"].tolist()
    values = profile_df["Percentile"].tolist()

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        name="Player"
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%"
            )
        ),
        height=520,
        margin=dict(l=50, r=50, t=80, b=40),
        showlegend=False
    )

    return fig


def get_cluster_profile(players_df, features, cluster):
    cluster_rows = players_df[players_df["Cluster"] == cluster]

    if cluster_rows.empty:
        return pd.DataFrame(columns=["Feature", "Average"])

    available = [f for f in features if f in cluster_rows.columns]
    averages = cluster_rows[available].apply(
        pd.to_numeric, errors="coerce"
    ).mean()

    return (
        averages.rename("Average")
        .reset_index()
        .rename(columns={"index": "Feature"})
        .sort_values("Average", ascending=False)
    )


def build_recommendations(position, player_name, players_df, priority_values, top_n):
    available_groups = priority_features[position]

    normalized = {}
    total = sum(priority_values.values())

    if total > 0:
        normalized = {
            group: value / total
            for group, value in priority_values.items()
        }
    else:
        normalized = {group: 0 for group in priority_values}

    similarity_scores = get_similarity_score(position, player_name)

    player_data = players_df.copy()

    player_data["Similarity"] = player_data["Player"].map(
        similarity_scores
    ).fillna(0)

    player_data = player_data[
        player_data["Player"].str.casefold() != str(player_name).casefold()
    ].copy()

    performance_score = pd.Series(
        0.0, index=player_data.index, dtype=float
    )

    for group, weight in normalized.items():
        if weight == 0:
            continue

        selected_features = [
            f for f in available_groups[group]
            if f in player_data.columns
        ]

        if not selected_features:
            continue

        ranks = player_data[selected_features].apply(
            pd.to_numeric, errors="coerce"
        ).rank(pct=True)

        group_score = ranks.mean(axis=1).fillna(0)
        performance_score += group_score * weight

    player_data["Performance Score"] = performance_score

    player_data["Scouting Score"] = (
        0.60 * player_data["Similarity"]
        + 0.40 * player_data["Performance Score"]
    )

    cols = [
        "Player", "Min", "90s", "Cluster",
        "Similarity", "Performance Score", "Scouting Score"
    ]

    recommendations = (
        player_data[cols]
        .sort_values("Scouting Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    for col in ["Similarity", "Performance Score", "Scouting Score"]:
        recommendations[col] = (
            pd.to_numeric(recommendations[col], errors="coerce")
            .fillna(0) * 100
        ).round(1)

    return recommendations


# ============================================================
# HEADER
# ============================================================

st.title("⚽ Football Intelligence Engine")

st.markdown(
    """
    **Data-driven player profiling, similarity and scouting engine**

    Explore player performance, statistical archetypes, similar players
    and configurable replacement recommendations.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎯 Player Selection")

position = st.sidebar.selectbox(
    "Position Group",
    list(dashboard_data.keys())
)

players_df = cluster_results[position]["data"].copy()
players_df["Player"] = players_df["Player"].astype(str)

player_list = sorted(players_df["Player"].dropna().unique().tolist())

if not player_list:
    st.error("No players are available for this position group.")
    st.stop()

player_name = st.sidebar.selectbox(
    "Player",
    player_list
)

top_n = st.sidebar.slider(
    "Recommendations",
    min_value=5,
    max_value=15,
    value=5
)

player_row = players_df[
    players_df["Player"].str.casefold() == player_name.casefold()
].iloc[0]

cluster = int(player_row["Cluster"])
features = dashboard_data[position]["features"]

# ============================================================
# TOP SUMMARY
# ============================================================

st.caption(
    f"Position group: **{position}**  •  Dataset players: **{len(players_df):,}**"
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Player", player_name)

with c2:
    st.metric("Minutes", f"{int(safe_float(player_row.get('Min', 0))):,}")

with c3:
    st.metric("90s Played", f"{safe_float(player_row.get('90s', 0)):.1f}")

with c4:
    st.metric("Archetype", f"Cluster {cluster}")

with c5:
    sim_preview = get_similarity_table(position, player_name)
    best_similarity = (
        sim_preview["Similarity"].iloc[0] * 100
        if not sim_preview.empty else 0
    )
    st.metric("Top Similarity", f"{best_similarity:.1f}%")

st.divider()

# ============================================================
# TABS
# ============================================================

tab_profile, tab_archetype, tab_similarity, tab_scouting = st.tabs(
    [
        "👤 Player Profile",
        "🧬 Archetype Explorer",
        "🔎 Similar Players",
        "🎯 Scouting Engine"
    ]
)

# ============================================================
# PLAYER PROFILE
# ============================================================

with tab_profile:
    st.subheader(f"📊 {player_name} — Performance Profile")

    profile_df = percentile_profile(
        players_df,
        features,
        player_row
    )

    top_profile = (
        profile_df.sort_values("Percentile", ascending=False)
        .head(10)
        .sort_values("Percentile")
    )

    left, right = st.columns([1.55, 1])

    with left:
        fig = px.bar(
            top_profile,
            x="Percentile",
            y="Feature",
            orientation="h",
            range_x=[0, 100],
            title="Top Performance Percentiles",
            labels={"Percentile": "Percentile"}
        )
        fig.update_layout(
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        radar_features = (
            profile_df.sort_values(
                "Percentile",
                ascending=False
            )
            .head(8)
            .sort_values("Feature")
        )

        radar = make_radar(
            radar_features,
            "Player Profile — Percentile View"
        )

        if radar:
            st.plotly_chart(radar, use_container_width=True)

    st.subheader("Metric Explorer")

    selected_metric = st.selectbox(
        "Choose a metric to compare",
        [f for f in features if f in players_df.columns]
    )

    metric_df = players_df[
        ["Player", selected_metric]
    ].copy()

    metric_df[selected_metric] = pd.to_numeric(
        metric_df[selected_metric],
        errors="coerce"
    )

    metric_df = metric_df.dropna().sort_values(
        selected_metric,
        ascending=False
    )

    player_metric = safe_float(
        player_row.get(selected_metric, 0)
    )

    percentile = (
        (metric_df[selected_metric] <= player_metric).mean() * 100
        if not metric_df.empty else 50
    )

    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            selected_metric,
            f"{player_metric:.2f}"
        )

    with m2:
        st.metric(
            "Position Percentile",
            f"{percentile:.1f}%"
        )

    st.dataframe(
        metric_df.head(15),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# ARCHETYPE EXPLORER
# ============================================================

with tab_archetype:

    st.subheader("🧬 Statistical Archetype Explorer")

    # --------------------------------------------------------
    # Cluster counts
    # --------------------------------------------------------

    cluster_counts = (
        players_df["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Players")
    )

    # --------------------------------------------------------
    # Convert numerical clusters into football archetypes
    # --------------------------------------------------------

    archetype_options = {
        cluster_id: get_archetype_name(position, cluster_id)
        for cluster_id in cluster_counts["Cluster"].tolist()
    }

    archetype_names = list(archetype_options.values())

    # Find the current player's archetype
    current_archetype = get_archetype_name(position, cluster)

    # Selectbox
    default_index = (
        archetype_names.index(current_archetype)
        if current_archetype in archetype_names
        else 0
    )

    selected_archetype = st.selectbox(
        "Explore player archetype",
        archetype_names,
        index=default_index
    )

    # Convert selected archetype back to numerical cluster
    selected_cluster = next(
        cluster_id
        for cluster_id, archetype_name in archetype_options.items()
        if archetype_name == selected_archetype
    )

    # --------------------------------------------------------
    # Number of players in selected archetype
    # --------------------------------------------------------

    cluster_size = int(
        cluster_counts.loc[
            cluster_counts["Cluster"] == selected_cluster,
            "Players"
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Player archetype information
    # --------------------------------------------------------

    archetype_name = get_archetype_name(
        position,
        selected_cluster
    )

    st.info(
        f"**{player_name}** fits the **{archetype_name}** profile. "
        f"This archetype contains **{cluster_size} players** "
        f"in the {position.lower()} dataset."
    )

    # --------------------------------------------------------
    # Cluster characteristics
    # --------------------------------------------------------

    cluster_profile = get_cluster_profile(
        players_df,
        features,
        selected_cluster
    )

    top_cluster = (
        cluster_profile
        .head(10)
        .sort_values("Average")
    )

    fig_cluster = px.bar(
        top_cluster,
        x="Average",
        y="Feature",
        orientation="h",
        title=f"{archetype_name} – Key Characteristics"
    )

    fig_cluster.update_layout(
        height=480,
        showlegend=False
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Players belonging to selected archetype
    # --------------------------------------------------------

    st.subheader("Players in this Archetype")

    archetype_players = (
        players_df[
            players_df["Cluster"] == selected_cluster
        ][["Player", "Min", "90s"]]
        .sort_values("Min", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        archetype_players,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Download archetype player list
    # --------------------------------------------------------

    csv = archetype_players.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Player List",
        data=csv,
        file_name=f"{archetype_name.replace(' ', '_').lower()}_players.csv",
        mime="text/csv"
    )

# ============================================================
# SIMILAR PLAYERS
# ============================================================

with tab_similarity:
    st.subheader(f"🔎 Players Similar to {player_name}")

    similarity_df = get_similarity_table(
        position,
        player_name
    )

    if similarity_df.empty:
        st.warning("Similarity data is unavailable for this player.")
    else:
        display_df = similarity_df.head(top_n).copy()
        display_df["Similarity"] = (
            display_df["Similarity"] * 100
        ).round(1)

        fig_similarity = px.bar(
            display_df.sort_values("Similarity"),
            x="Similarity",
            y="Player",
            orientation="h",
            range_x=[0, 100],
            title="Similarity Ranking",
            labels={"Similarity": "Similarity (%)"}
        )

        fig_similarity.update_layout(
            height=430,
            showlegend=False
        )

        st.plotly_chart(
            fig_similarity,
            use_container_width=True
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Player Comparison")

        comparison_options = [
            p for p in player_list
            if p.casefold() != player_name.casefold()
        ]

        if comparison_options:
            compare_player = st.selectbox(
                "Compare with another player",
                comparison_options
            )

            compare_row = players_df[
                players_df["Player"].str.casefold()
                == compare_player.casefold()
            ].iloc[0]

            compare_features = [
                f for f in features
                if f in players_df.columns
            ]

            p1 = percentile_profile(
                players_df,
                compare_features,
                player_row
            ).set_index("Feature")

            p2 = percentile_profile(
                players_df,
                compare_features,
                compare_row
            ).set_index("Feature")

            comparison = pd.DataFrame({
                player_name: p1["Percentile"],
                compare_player: p2["Percentile"]
            }).fillna(50)

            comparison = (
                comparison
                .sort_values(player_name, ascending=False)
                .head(12)
                .sort_values(player_name)
                .reset_index()
                .rename(columns={"index": "Feature"})
            )

            fig_compare = px.bar(
                comparison,
                x="Feature",
                y=[player_name, compare_player],
                barmode="group",
                title=f"{player_name} vs {compare_player} — Percentile Comparison"
            )

            fig_compare.update_layout(
                height=500,
                yaxis_title="Percentile"
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

# ============================================================
# SCOUTING / REPLACEMENT ENGINE
# ============================================================

with tab_scouting:

    st.subheader("🎯 Configurable Scouting / Replacement Engine")

    st.write(
        "Adjust the importance of different football attributes. "
        "The engine combines **player similarity** with "
        "**performance fit** to rank potential replacements."
    )

    # --------------------------------------------------------
    # Available scouting priorities for this position
    # --------------------------------------------------------

    available_groups = priority_features[position]

    priority_values = {}

    # Keep sliders in compact rows
    groups = list(available_groups.keys())

    for start in range(0, len(groups), 3):

        row_groups = groups[start:start + 3]
        cols = st.columns(len(row_groups))

        for col, group in zip(cols, row_groups):

            with col:

                priority_values[group] = st.slider(
                    group,
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    key=f"priority_{position}_{group}"
                )

    # --------------------------------------------------------
    # Priority summary
    # --------------------------------------------------------

    total_priority = sum(priority_values.values())

    if total_priority == 0:

        st.warning(
            "All priorities are currently zero. "
            "The performance component will contribute zero."
        )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    if st.button(
        "🔎 Find Best Replacement Players",
        type="primary",
        use_container_width=True
    ):

        recommendations = build_recommendations(
            position,
            player_name,
            players_df,
            priority_values,
            top_n
        )

        # Store results in session state
        st.session_state[
            f"recommendations_{position}"
        ] = recommendations

        st.session_state[
            f"recommendation_target_{position}"
        ] = player_name

    # --------------------------------------------------------
    # Display recommendations if available
    # --------------------------------------------------------

    recommendation_key = f"recommendations_{position}"
    target_key = f"recommendation_target_{position}"

    if recommendation_key in st.session_state:

        recommendations = st.session_state[recommendation_key]
        target_player = st.session_state[target_key]

        st.success(
            f"Scouting report generated for **{target_player}**."
        )

        # ----------------------------------------------------
        # Top recommendation
        # ----------------------------------------------------

        if len(recommendations) > 0:

            best_player = recommendations.iloc[0]["Player"]

            best_score = None

            if "Scouting Score" in recommendations.columns:
                best_score = recommendations.iloc[0]["Scouting Score"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "🥇 Top Replacement",
                    best_player
                )

            with col2:

                if best_score is not None:
                    st.metric(
                        "Scouting Score",
                        f"{best_score:.1f}"
                    )
                else:
                    st.metric(
                        "Candidates Found",
                        len(recommendations)
                    )

        # ----------------------------------------------------
        # Recommendation table
        # ----------------------------------------------------

        st.subheader("🏆 Recommended Replacements")

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Interactive ranking chart
        # ----------------------------------------------------

        if (
            "Player" in recommendations.columns
            and "Scouting Score" in recommendations.columns
        ):

            chart_df = (
                recommendations[
                    ["Player", "Scouting Score"]
                ]
                .sort_values(
                    "Scouting Score",
                    ascending=True
                )
            )

            fig_recommendations = px.bar(
                chart_df,
                x="Scouting Score",
                y="Player",
                orientation="h",
                range_x=[0, 100],
                title="Replacement Candidate Ranking",
                labels={
                    "Scouting Score": "Scouting Score (%)",
                    "Player": "Player"
                }
            )

            fig_recommendations.update_layout(
                height=max(
                    450,
                    len(chart_df) * 55
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig_recommendations,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Download scouting report
        # ----------------------------------------------------

        report_csv = recommendations.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Scouting Report",
            data=report_csv,
            file_name=(
                f"{target_player.replace(' ', '_')}"
                f"_replacement_report.csv"
            ),
            mime="text/csv"
        )

        # ----------------------------------------------------
        # Explain the ranking
        # ----------------------------------------------------

        st.subheader("ℹ️ How the Ranking Works")

        st.markdown(
            """
            **Scouting Score = 60% Player Similarity + 40% Performance Fit**

            **Similarity** measures how statistically close the candidate
            is to the selected player based on the model's football
            performance features.

            **Performance Fit** evaluates candidates within the selected
            position according to the priorities chosen above.

            Use the sliders to change what matters most in the scouting
            decision.
            """
        )

    else:

        st.info(
            "Adjust the priorities above and click "
            "**Find Best Replacement Players** to generate "
            "your scouting shortlist."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Football Intelligence Engine | "
    "Python • Pandas • Scikit-learn • Plotly • Streamlit"
)
