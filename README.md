# Earthquake Building Damage Grade Prediction
Multiclass classification model to predict earthquake building damage grade using Random Forest and Streamlit deployment

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/farhanhl-ds/earthquake-damage-predictor)

## Repository Outline

```
1. modeling.ipynb      - Main modeling notebook: EDA, feature engineering, model training, evaluation, and saving
2. inference.ipynb     - Model inference notebook: single and batch prediction examples
3. conceptual.txt      - Answers to conceptual questions on bagging, Random Forest, and cross validation
4. url.txt             - Links to dataset, saved model, and deployment
5. deployment/
   ├── src/
   │   ├── streamlit_app.py              - Streamlit web application
   │   ├── nepal_earthquake.jpg          - Hero image for the app
   │   ├── label_encoder.pkl             - Label encoder for target variable
   │   ├── best_model_earthquake.pkl     - ⚠️ Not included (347 MB) → see url.txt
   │   └── csv_building_structure.csv    - ⚠️ Not included (155 MB) → see url.txt
   ├── Dockerfile
   ├── .gitattributes
   ├── README.md
   └── requirements.txt
```

> ⚠️ Large files (`best_model_earthquake.pkl` and `csv_building_structure.csv`) are not tracked in this repository. Download links are available in `url.txt`.

## Problem Background

On April 25, 2015, a 7.8 Mw earthquake struck Nepal, claiming over 8,700 lives and destroying hundreds of thousands of buildings across 31 districts. In the aftermath, the Government of Nepal conducted one of the largest post-disaster building surveys ever recorded, covering approximately 762,000 structures across 11 severely affected districts.

A key challenge in disaster response is the rapid and accurate assessment of building damage. Traditional field surveys are time-consuming and resource-intensive, while misclassification of damage levels, especially underestimating severe damage can result in inefficient allocation of emergency resources and delayed reconstruction.

This project builds a machine learning model that predicts the **damage grade** of a building (Grade 1 to 5) based on its structural characteristics recorded *before* the earthquake. The model can support disaster response agencies such as BNPB and BPBD in Indonesia to rapidly prioritize areas for emergency response, estimate structural losses, and inform earthquake-resistant construction standards, particularly relevant given Indonesia's high seismic vulnerability.

## Project Output

A trained **Random Forest** multiclass classification model saved as `best_model_earthquake.pkl`, capable of predicting building damage grade (1–5) from structural input features. The model is served through a Streamlit web application that accepts both single and batch predictions.

## Data

- **Source**: [Earthquake Magnitude, Damage and Impact (Kaggle)](https://www.kaggle.com/datasets/arashnic/earthquake-magnitude-damage-and-impact)
- **File used**: `csv_building_structure.csv`
- **Original size**: ~762,000 rows and 40 columns
- **Sampled size**: 50,000 rows (stratified random sample for training efficiency)
- **Features**: Mix of numerical (building age, floor count, height, plinth area) and categorical (foundation type, roof type, superstructure material, land surface condition, etc.)
- **Target**: `damage_grade` - ordinal label from Grade 1 (negligible damage) to Grade 5 (complete destruction)
- **Class distribution**: Imbalanced - Grade 5 dominates (~36%), Grade 1 is the smallest class (~10%)
- **Missing values**: None after column dropping
- **Columns dropped**: Identifier columns (`building_id`, `district_id`, etc.) and post-earthquake columns that would cause data leakage (`count_floors_post_eq`, `condition_post_eq`, `technical_solution_proposed`)

## Method

This project implements a **Supervised Learning - Multiclass Classification** pipeline with the following steps:

1. **Exploratory Data Analysis (EDA)**: Distribution analysis, feature-target relationships, correlation heatmap, and superstructure material analysis
2. **Feature Engineering**: Column dropping (leakage prevention), label encoding of target, train-test split (80:20 stratified), and a `ColumnTransformer` preprocessing pipeline (StandardScaler for numerical, OneHotEncoder for categorical, passthrough for binary)
3. **Model Training & Cross Validation**: Five baseline models evaluated with 5-Fold Stratified Cross Validation using F1-Weighted as the primary metric- KNN, SVM, Decision Tree, Random Forest, and Gradient Boosting
4. **Model Selection**: Random Forest selected based on the best F1-Weighted score and smallest Micro-Macro F1 gap, indicating the most balanced performance across all damage grades
5. **Hyperparameter Tuning**: RandomizedSearchCV applied to Random Forest with 20 iterations and 5-fold CV
6. **Evaluation**: Confusion matrix, classification report, and before/after tuning performance comparison

**Primary evaluation metric**: F1-Weighted (chosen over accuracy due to class imbalance)

## Stacks

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.10 |
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Resampling | `imbalanced-learn` |
| Model Saving | `pickle` |
| Deployment | `streamlit` |
| Environment | Jupyter Notebook |

## Reference

- **Dataset**: https://www.kaggle.com/datasets/arashnic/earthquake-magnitude-damage-and-impact
- **Saved Model (Google Drive)**: *https://drive.google.com/drive/folders/1muXQXor_Zwul_3Vhc54OVH2_giEBxDfr?usp=sharing*
- **Deployment (Streamlit)**: *https://huggingface.co/spaces/farhanhl-ds/earthquake-damage-predictor*
- DrivenData Competition - Richter's Predictor: https://www.drivendata.org/competitions/57/nepal-earthquake/
- Nepal Earthquake 2015 - National Planning Commission: https://www.npc.gov.np/
- BNPB Indonesia Earthquake Risk: https://www.bnpb.go.id/