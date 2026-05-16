import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Path helper (works both locally and in HuggingFace src/ subfolder) ────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def p(filename):
    return os.path.join(BASE_DIR, filename)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earthquake Damage Predictor",
    page_icon="🏚️",
    layout="wide"
)

# ── Load model & label encoder ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load(p('best_model_earthquake.pkl'))
    le    = joblib.load(p('label_encoder.pkl'))
    return model, le

model, le = load_artifacts()

# ── Load dataset for EDA ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(p('csv_building_structure.csv'))
    df = df.dropna(subset=['damage_grade'])
    df['damage_grade'] = df['damage_grade'].astype(str)
    return df.sample(n=50000, random_state=7).reset_index(drop=True)

# ── Grade metadata ────────────────────────────────────────────────────────────
GRADE_INFO = {
    'Grade 1': {'label': 'Negligible to Slight Damage',  'color': '#2ecc71', 'emoji': '🟢'},
    'Grade 2': {'label': 'Moderate Damage',               'color': '#f1c40f', 'emoji': '🟡'},
    'Grade 3': {'label': 'Substantial to Heavy Damage',   'color': '#e67e22', 'emoji': '🟠'},
    'Grade 4': {'label': 'Very Heavy Damage',             'color': '#e74c3c', 'emoji': '🔴'},
    'Grade 5': {'label': 'Complete Destruction',          'color': '#8e44ad', 'emoji': '🟣'},
}
GRADE_COLORS = [GRADE_INFO[g]['color'] for g in sorted(GRADE_INFO.keys())]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏚️ Damage Predictor")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA", "🔍 Single Prediction", "📋 Batch Prediction"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model**: Random Forest  \n"
    "**Metric**: F1-Weighted  \n"
    "**Dataset**: Nepal Earthquake 2015  \n"
    "**Classes**: Grade 1 to 5"
)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏚️ Earthquake Building Damage Grade Predictor")
    st.markdown("#### Predict structural damage levels based on pre-earthquake building characteristics")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Random Forest")
    c2.metric("Primary Metric", "F1-Weighted")
    c3.metric("Damage Classes", "5 Grades")

    st.markdown("---")
    col_img, col_about = st.columns([1, 2])
    with col_img:
        st.image(
            p("nepal_earthquake.jpg"),
            caption="Aftermath of the 2015 Gorkha Earthquake, Nepal | picture-alliance/dpa/AP",
            width=300
        )
    with col_about:
        st.markdown("### About This App")
        st.markdown(
            "This application uses a machine learning model trained on post-disaster building "
            "survey data from the **2015 Gorkha Earthquake in Nepal** (covering ~762,000 buildings) "
            "to predict the **damage grade** of a building based on its structural characteristics "
            "recorded *before* the earthquake."
        )
        st.markdown("**Damage Grade Scale:**")
        for grade, info in GRADE_INFO.items():
            st.markdown(info["emoji"] + " **" + grade + "** — " + info["label"])

    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown(
        "- **📊 EDA** — Explore the dataset: class distribution, feature patterns, and relationships with damage grade.\n"
        "- **🔍 Single Prediction** — Input one building's characteristics for an instant prediction with probability breakdown.\n"
        "- **📋 Batch Prediction** — Upload a CSV file to predict multiple buildings at once."
    )
    st.caption(
        "⚠️ This model was trained on Nepal building data. Results are indicative and should not "
        "be used as the sole basis for decisions without local validation."
    )


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Exploring patterns in the Nepal earthquake building dataset that drive structural damage prediction.")
    st.markdown("---")

    with st.spinner("Loading dataset..."):
        df = load_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Target Distribution",
        "🔢 Numerical Features",
        "🏷️ Categorical Features",
        "🔗 Feature vs Target",
    ])

    with tab1:
        st.markdown("### Damage Grade Distribution")
        st.markdown(
            "The target variable `damage_grade` is **imbalanced**: Grade 5 dominates at ~36% "
            "while Grade 1 is the smallest class at ~10%."
        )
        grade_counts = df['damage_grade'].value_counts().reindex(sorted(df['damage_grade'].unique()))
        grade_pct    = (grade_counts / grade_counts.sum() * 100).round(1)
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=grade_counts.index, y=grade_counts.values,
                marker_color=GRADE_COLORS,
                text=[f"{v:,} ({p}%)" for v, p in zip(grade_counts.values, grade_pct.values)],
                textposition='outside'
            ))
            fig.update_layout(title="Count per Damage Grade", xaxis_title="Damage Grade",
                              yaxis_title="Number of Buildings", height=420, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Pie(
                labels=grade_counts.index, values=grade_counts.values,
                marker_colors=GRADE_COLORS, hole=0.4, textinfo='label+percent'
            ))
            fig.update_layout(title="Proportion of Each Grade", height=420, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        st.info(
            "💡 A naive model that always predicts Grade 5 would achieve ~36% accuracy without learning anything. "
            "**F1-Weighted** is used as the primary metric to account for class imbalance."
        )

    with tab2:
        st.markdown("### Numerical Features Distribution")
        st.markdown("All four numerical features show a **right-skewed** distribution with outliers.")
        num_cols = ['count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq']
        num_labels = {
            'count_floors_pre_eq': 'Number of Floors',
            'age_building':        'Building Age (years)',
            'plinth_area_sq_ft':   'Plinth Area (sq ft)',
            'height_ft_pre_eq':    'Height in ft',
        }
        col1, col2 = st.columns(2)
        for idx, col in enumerate(num_cols):
            target = col1 if idx % 2 == 0 else col2
            fig = px.histogram(df, x=col, nbins=40, title=num_labels[col],
                               color_discrete_sequence=['#3498db'])
            fig.update_layout(height=300, margin=dict(t=40, b=20), showlegend=False)
            target.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df[num_cols].rename(columns=num_labels).describe().round(2), use_container_width=True)
        st.info(
            "💡 Highly skewed distributions favour **tree-based models** (Random Forest, Gradient Boosting) "
            "over distance-based models (KNN, SVM). Tree models split on thresholds — outliers do not distort the model."
        )
        st.markdown("#### Correlation Heatmap")
        corr = df[num_cols].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=[num_labels[c] for c in num_cols],
            y=[num_labels[c] for c in num_cols],
            colorscale='RdBu_r', zmid=0,
            text=corr.values, texttemplate="%{text}", textfont={"size": 13}
        ))
        fig.update_layout(title="Correlation Matrix — Numerical Features", height=420, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Categorical Features Distribution")
        cat_cols = {
            'foundation_type':        'Foundation Type',
            'roof_type':              'Roof Type',
            'land_surface_condition': 'Land Surface Condition',
            'ground_floor_type':      'Ground Floor Type',
            'other_floor_type':       'Other Floor Type',
            'position':               'Building Position',
            'plan_configuration':     'Plan Configuration',
        }
        selected_label = st.selectbox("Choose a categorical feature", list(cat_cols.values()))
        selected_col   = [k for k, v in cat_cols.items() if v == selected_label][0]
        counts = df[selected_col].value_counts().reset_index()
        counts.columns = [selected_label, 'Count']
        counts['Pct'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
        fig = px.bar(
            counts, x='Count', y=selected_label, orientation='h',
            text=counts['Pct'].astype(str) + '%',
            color='Count', color_continuous_scale='Blues',
            title=f"Distribution of {selected_label}"
        )
        fig.update_layout(height=420, margin=dict(t=50, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Superstructure Material Usage")
        super_cols = {
            'has_superstructure_adobe_mud':           'Adobe Mud',
            'has_superstructure_mud_mortar_stone':    'Mud Mortar Stone',
            'has_superstructure_stone_flag':          'Stone Flag',
            'has_superstructure_cement_mortar_stone': 'Cement Mortar Stone',
            'has_superstructure_mud_mortar_brick':    'Mud Mortar Brick',
            'has_superstructure_cement_mortar_brick': 'Cement Mortar Brick',
            'has_superstructure_timber':              'Timber',
            'has_superstructure_bamboo':              'Bamboo',
            'has_superstructure_rc_non_engineered':   'RC Non-Engineered',
            'has_superstructure_rc_engineered':       'RC Engineered',
            'has_superstructure_other':               'Other',
        }
        usage = df[list(super_cols.keys())].mean().rename(super_cols) * 100
        usage = usage.sort_values(ascending=True)
        fig = px.bar(
            x=usage.values, y=usage.index, orientation='h',
            text=[f"{v:.1f}%" for v in usage.values],
            color=usage.values, color_continuous_scale='Oranges',
            title="Superstructure Material Usage (%)"
        )
        fig.update_layout(height=400, margin=dict(t=50, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Feature Relationships with Damage Grade")
        sub1, sub2, sub3, sub4 = st.tabs([
            "Age vs Grade", "Foundation vs Grade", "Superstructure vs Grade", "Floors vs Grade"
        ])
        with sub1:
            fig = px.box(
                df, x='damage_grade', y='age_building',
                color='damage_grade', color_discrete_sequence=GRADE_COLORS,
                category_orders={'damage_grade': sorted(df['damage_grade'].unique())},
                labels={'damage_grade': 'Damage Grade', 'age_building': 'Building Age (years)'},
                title="Building Age Distribution per Damage Grade"
            )
            fig.update_layout(showlegend=False, height=440)
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Median building age increases consistently from Grade 1 to Grade 5. **Older buildings are significantly more vulnerable** to earthquake damage.")
        with sub2:
            ct = pd.crosstab(df['foundation_type'], df['damage_grade'], normalize='index') * 100
            ct = ct.reindex(columns=sorted(ct.columns))
            fig = go.Figure()
            for col, color in zip(ct.columns, GRADE_COLORS):
                fig.add_trace(go.Bar(
                    name=col, x=ct.index, y=ct[col], marker_color=color,
                    text=ct[col].round(1).astype(str) + '%', textposition='inside'
                ))
            fig.update_layout(
                barmode='stack', title="Foundation Type vs Damage Grade (%)",
                xaxis_title="Foundation Type", yaxis_title="Percentage (%)",
                height=450, legend_title="Damage Grade", margin=dict(t=50)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Mud mortar stone/brick foundations show much higher Grade 4-5 damage. **RC foundations** correlate strongly with lower damage grades.")
        with sub3:
            s_cols = {
                'has_superstructure_adobe_mud':           'Adobe Mud',
                'has_superstructure_mud_mortar_stone':    'Mud Mortar Stone',
                'has_superstructure_cement_mortar_brick': 'Cement Mortar Brick',
                'has_superstructure_timber':              'Timber',
                'has_superstructure_bamboo':              'Bamboo',
                'has_superstructure_rc_engineered':       'RC Engineered',
            }
            sbg = df.groupby('damage_grade')[list(s_cols.keys())].mean()
            sbg.columns = list(s_cols.values())
            sbg = sbg.reindex(sorted(sbg.index))
            melted = sbg.T.reset_index().melt(id_vars='index')
            fig = px.bar(
                melted, x='index', y='value', color='damage_grade',
                barmode='group', color_discrete_sequence=GRADE_COLORS,
                labels={'index': 'Material', 'value': 'Avg Usage Proportion', 'damage_grade': 'Damage Grade'},
                title="Average Superstructure Material Usage per Damage Grade"
            )
            fig.update_layout(height=450, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Adobe mud and mud mortar stone dominate Grade 4-5. **RC engineered** construction is more common in Grade 1-2.")
        with sub4:
            floor_grade = df.groupby(['damage_grade', 'count_floors_pre_eq']).size().reset_index(name='count')
            floor_grade = floor_grade[floor_grade['count_floors_pre_eq'] <= 5]
            fig = px.bar(
                floor_grade, x='count_floors_pre_eq', y='count', color='damage_grade',
                barmode='group', color_discrete_sequence=GRADE_COLORS,
                labels={'count_floors_pre_eq': 'Number of Floors', 'count': 'Number of Buildings', 'damage_grade': 'Damage Grade'},
                title="Building Count by Floor Number and Damage Grade"
            )
            fig.update_layout(height=450, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 As floor count increases, the proportion of higher damage grades rises due to greater structural stress during seismic events.")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Prediction":
    st.title("🔍 Single Building Prediction")
    st.markdown("Fill in the structural characteristics of the building below.")
    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown("#### 🔢 Numerical Features")
        col1, col2, col3, col4 = st.columns(4)
        count_floors   = col1.number_input("Number of Floors (pre-earthquake)", min_value=1, max_value=20, value=2)
        age_building   = col2.number_input("Building Age (years)", min_value=0, max_value=200, value=25)
        plinth_area    = col3.number_input("Plinth Area (sq ft)", min_value=10, max_value=10000, value=400)
        height_ft      = col4.number_input("Height (ft, pre-earthquake)", min_value=5.0, max_value=100.0, value=12.0, step=0.5)

        st.markdown("#### 🏷️ Categorical Features")
        col1, col2, col3, col4 = st.columns(4)
        land_surface   = col1.selectbox("Land Surface Condition", ['Flat', 'Moderate slope', 'Steep slope'])
        foundation     = col2.selectbox("Foundation Type", ['Mud mortar-Stone/Brick', 'Cement-Stone/Brick', 'RC', 'Bamboo/Timber', 'Other'])
        roof           = col3.selectbox("Roof Type", ['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof', 'RCC/RB/RBC'])
        ground_floor   = col4.selectbox("Ground Floor Type", ['Mud', 'Brick/Stone', 'RC', 'Timber', 'Other'])

        col1, col2, col3 = st.columns(3)
        other_floor    = col1.selectbox("Other Floor Type", ['Not applicable', 'TImber/Bamboo-Mud', 'Timber-Planck', 'RCC/RB/RBC'])
        position       = col2.selectbox("Position", ['Not attached', 'Attached-1 side', 'Attached-2 side', 'Attached-3 side'])
        plan_config    = col3.selectbox("Plan Configuration", ['Rectangular', 'L-shape', 'T-shape', 'E-shape', 'U-shape', 'H-shape', 'Multi-projected', 'Others', 'Square', 'Building with Central Courtyard'])

        st.markdown("#### 🧱 Superstructure Materials")
        st.caption("Select all materials used in the building's superstructure")
        col1, col2, col3, col4 = st.columns(4)
        adobe_mud           = int(col1.checkbox("Adobe Mud"))
        mud_mortar_stone    = int(col1.checkbox("Mud Mortar Stone"))
        stone_flag          = int(col2.checkbox("Stone Flag"))
        cement_mortar_stone = int(col2.checkbox("Cement Mortar Stone"))
        mud_mortar_brick    = int(col3.checkbox("Mud Mortar Brick"))
        cement_mortar_brick = int(col3.checkbox("Cement Mortar Brick"))
        timber              = int(col4.checkbox("Timber"))
        bamboo              = int(col4.checkbox("Bamboo"))

        col1, col2, col3 = st.columns(3)
        rc_non_engineered   = int(col1.checkbox("RC Non-Engineered"))
        rc_engineered       = int(col2.checkbox("RC Engineered"))
        other_material      = int(col3.checkbox("Other Material"))

        submitted = st.form_submit_button("🔮 Predict Damage Grade", use_container_width=True)

    if submitted:
        input_data = {
            'count_floors_pre_eq': count_floors, 'age_building': age_building,
            'plinth_area_sq_ft': plinth_area, 'height_ft_pre_eq': height_ft,
            'land_surface_condition': land_surface, 'foundation_type': foundation,
            'roof_type': roof, 'ground_floor_type': ground_floor,
            'other_floor_type': other_floor, 'position': position,
            'plan_configuration': plan_config,
            'has_superstructure_adobe_mud': adobe_mud,
            'has_superstructure_mud_mortar_stone': mud_mortar_stone,
            'has_superstructure_stone_flag': stone_flag,
            'has_superstructure_cement_mortar_stone': cement_mortar_stone,
            'has_superstructure_mud_mortar_brick': mud_mortar_brick,
            'has_superstructure_cement_mortar_brick': cement_mortar_brick,
            'has_superstructure_timber': timber,
            'has_superstructure_bamboo': bamboo,
            'has_superstructure_rc_non_engineered': rc_non_engineered,
            'has_superstructure_rc_engineered': rc_engineered,
            'has_superstructure_other': other_material,
        }
        df_input     = pd.DataFrame([input_data])
        pred_encoded = model.predict(df_input)
        pred_label   = le.inverse_transform(pred_encoded)[0]
        pred_proba   = model.predict_proba(df_input)[0]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        info = GRADE_INFO[pred_label]
        st.markdown(
            f"<div style='background-color:{info['color']}22; border-left:6px solid {info['color']}; "
            f"padding:20px; border-radius:8px;'>"
            f"<h2 style='color:{info['color']}; margin:0'>{info['emoji']} {pred_label}</h2>"
            f"<p style='margin:4px 0 0 0; font-size:18px'>{info['label']}</p></div>",
            unsafe_allow_html=True
        )
        st.markdown("#### 📊 Prediction Probabilities")
        fig = go.Figure(go.Bar(
            x=le.classes_, y=pred_proba,
            marker_color=[GRADE_INFO[g]['color'] for g in le.classes_],
            text=[f"{p:.1%}" for p in pred_proba], textposition='outside'
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 1], title="Probability"),
            xaxis_title="Damage Grade", height=380, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Prediction":
    st.title("📋 Batch Prediction")
    st.markdown("Upload a CSV file containing multiple building records to predict all damage grades at once.")
    st.markdown("---")

    template_data = {
        'count_floors_pre_eq': [2, 1, 3], 'age_building': [30, 80, 5],
        'plinth_area_sq_ft': [400, 250, 900], 'height_ft_pre_eq': [12.0, 8.0, 24.0],
        'land_surface_condition': ['Moderate slope', 'Steep slope', 'Flat'],
        'foundation_type': ['Mud mortar-Stone/Brick', 'Mud mortar-Stone/Brick', 'RC'],
        'roof_type': ['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof', 'RCC/RB/RBC'],
        'ground_floor_type': ['Mud', 'Mud', 'RC'],
        'other_floor_type': ['TImber/Bamboo-Mud', 'Not applicable', 'RCC/RB/RBC'],
        'position': ['Not attached', 'Attached-2 side', 'Not attached'],
        'plan_configuration': ['Rectangular', 'Rectangular', 'Rectangular'],
        'has_superstructure_adobe_mud': [0, 1, 0],
        'has_superstructure_mud_mortar_stone': [1, 1, 0],
        'has_superstructure_stone_flag': [0, 0, 0],
        'has_superstructure_cement_mortar_stone': [0, 0, 0],
        'has_superstructure_mud_mortar_brick': [0, 0, 0],
        'has_superstructure_cement_mortar_brick': [0, 0, 0],
        'has_superstructure_timber': [1, 0, 0],
        'has_superstructure_bamboo': [0, 0, 0],
        'has_superstructure_rc_non_engineered': [0, 0, 0],
        'has_superstructure_rc_engineered': [0, 0, 1],
        'has_superstructure_other': [0, 0, 0],
    }
    template_df = pd.DataFrame(template_data)
    st.download_button(
        label="📥 Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="building_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded {len(df_upload)} building records.**")
        st.dataframe(df_upload.head(), use_container_width=True)

        required_cols = list(template_data.keys())
        missing_cols  = [c for c in required_cols if c not in df_upload.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            with st.spinner("Running predictions..."):
                pred_encoded = model.predict(df_upload[required_cols])
                pred_labels  = le.inverse_transform(pred_encoded)
                pred_proba   = model.predict_proba(df_upload[required_cols])

            results = df_upload.copy()
            results.insert(0, 'Predicted Grade', pred_labels)
            for i, cls in enumerate(le.classes_):
                results[f'P({cls})'] = pred_proba[:, i].round(3)

            st.markdown("---")
            st.markdown("### 📊 Results")
            st.dataframe(results, use_container_width=True)

            grade_counts = pd.Series(pred_labels).value_counts().reindex(le.classes_, fill_value=0)
            fig = px.bar(
                x=grade_counts.index, y=grade_counts.values,
                color=grade_counts.index,
                color_discrete_map={g: GRADE_INFO[g]['color'] for g in GRADE_INFO},
                labels={'x': 'Damage Grade', 'y': 'Count'},
                title="Predicted Damage Grade Distribution"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                label="📥 Download Results as CSV",
                data=results.to_csv(index=False),
                file_name="prediction_results.csv",
                mime="text/csv"
            )