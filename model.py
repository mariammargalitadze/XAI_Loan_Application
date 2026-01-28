# model.py
import pandas as pd
import numpy as np
import shap
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        descriptive_cols = [
            'Portfolio Status', 'Turnover (Months)', 'Cash Transaction', 'Collateral', 
            'Residency', 'Gambling Records', 'Market Size', 'Interest Rate', 'Location', 
            'Loan Purpose', 'Yearly Growth Rate', 'Prior Default', 'Existing Liabilities', 
            'Active Years', 'Loan Maturity', 'Approved'
        ]
        df.columns = descriptive_cols
        
        # Cleaning
        df.replace('?', np.nan, inplace=True)
        cols_to_float = ['Turnover (Months)', 'Cash Transaction', 'Interest Rate', 'Active Years']
        cols_to_int = ['Yearly Growth Rate', 'Loan Maturity']
        
        for col in cols_to_float:
            df[col] = df[col].astype('float')
        for col in cols_to_int:
            df[col] = df[col].astype('int64')

        # Target encoding
        df['Approved'] = df['Approved'].replace({'+': 1, '-': 0})
        
        X = df.drop('Approved', axis=1)
        y = df['Approved']
        return df, X, y
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def train_pipeline(X, y):
    # Separating columns
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Pipeline
    #svm_pipeline = Pipeline(steps=[
    #    ('preprocessor', preprocessor),
    #    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
    #])
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_esimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    #svm_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)
    
    #return svm_pipeline, X_train, X_test, y_train, y_test
    return xgb_pipeline, X_train, X_test, y_train, y_test

#@st.cache_resource
#def setup_shap(_model, X_train):
#    def svm_pipeline_predict_proba(X_data):
#        if not isinstance(X_data, pd.DataFrame):
#            X_df = pd.DataFrame(X_data, columns=X_train.columns)
#        else:
#            X_df = X_data
#        return _model.predict_proba(X_df)[:, 1]
#
#    # Background data (subset for speed)
#    background_data = X_train.sample(100, random_state=42)
#    
#    # Initialize KernelExplainer
#    explainer = shap.KernelExplainer(svm_pipeline_predict_proba, background_data)
#    return explainer

@st.cache_resource
def setup_shap(_model, X_train):
    """
    TreeExplainer is much faster and more accurate for XGBoost.
    """
    # 1. Extract the fitted model from the pipeline
    model_inner = _model.named_steps['classifier']
    
    # 2. Extract the preprocessor to transform data before explaining
    # TreeExplainer needs the data in the format the model actually sees (encoded/scaled)
    preprocessor = _model.named_steps['preprocessor']
    X_train_transformed = preprocessor.transform(X_train)
    
    # 3. Use TreeExplainer
    explainer = shap.TreeExplainer(model_inner)
    
    # We store the transformed column names so the explanation plots look correct
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
    num_names = X_train.select_dtypes(include=np.number).columns.tolist()
    feature_names = np.concatenate([num_names, cat_names])
    
    return explainer, feature_names


# To enhance interpretability and user experience—central goals of an XAI dashboard—I renamed the anonymized features with meaningful labels. 
# This was achieved using a Python dictionary (rename_dict) and the pandas .rename() method. 
# The new feature names were informed by user stories and prior research on loan decision criteria, ensuring that the dataset became not only technically usable but also practically interpretable for underwriters. 
# Renaming served as a critical bridge between data analysis and the final interactive web application: without it, the system would have been functional at a computational level but largely unusable to stakeholders unfamiliar with the original dataset documentation. 