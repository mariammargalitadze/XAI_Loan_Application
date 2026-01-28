import streamlit as st
import model
import numpy as np

st.set_page_config(page_title="Introduction", layout="wide")

st.title("🏦 Loan Approval AI System")

if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0


# Initialization
if 'pipeline' not in st.session_state:
    with st.spinner("Loading Data and Training XGBoost Model..."):
        df, X, y = model.load_data("credit_approval_data.csv")
        
        if df is not None:
            # 1. Train the XGBoost Pipeline
            xgb_pipeline, X_train, X_test, y_train, y_test = model.train_pipeline(X, y)
            
            # 2. Setup the TreeExplainer and get the transformed feature names
            explainer, transformed_feature_names = model.setup_shap(xgb_pipeline, X_train)

            # 3. Create the 'train_df' (Combine X_train and y_train)
            train_df = X_train.copy()
            train_df['Approved'] = y_train # Use the exact name your CSV uses
            st.session_state['train_df'] = train_df

            # 4. Identify Continuous Columns
            # We filter only numeric columns and exclude the target
            st.session_state['continuous_cols'] = X_train.select_dtypes(include=['number']).columns.tolist()

            # Save to Session State
            st.session_state['pipeline'] = xgb_pipeline
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['explainer'] = explainer
            st.session_state['X'] = X 
            st.session_state['feature_names'] = transformed_feature_names
            st.session_state['selected_index'] = 0

            

            st.success("XGBoost Model trained and explainer ready!")
        else:
            st.error("Dataset not found. Please check the file path in model.py.")

  
st.divider()
st.markdown("<br>", unsafe_allow_html=True)

st.write('''
    Welcome to the XAI Streamlit Dashboard!\n\n
    This application uses **XGBoost** ML model for high-performance predictions and several **XAI methods** for the result explanation.\n\n
    ''')
st.write("By identifying trends from previous lending choices, XGBoost is a clever decision algorithm that assists in categorizing loan applications as accepted or denied.")
st.markdown("<br>", unsafe_allow_html=True)
st.write('''
    
    Navigate from the sidebar to explore answers to different quetions:
    * **Model:** Model insights and performance metrics, 
    * **Loan Decision:** Why specific decision? Who is similar? What-if analysis? What influences most? and Final Decision.
    * **Data Audit:** Checking data quality issues.
    ''')

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.info("Data used in this application comes from University of California, Irvine (UCI) Machine Learning Repository: [Credit Approval Data Set](https://archive.ics.uci.edu/dataset/27/credit+approval)")

st.write("")
# Academic Signature
st.divider()
st.caption("🎓 **Master of Data-driven Design (MDD)**")
st.caption("University of Applied Sciences Utrecht (HU)")
st.caption("Author: Mariam Margalitadze | Jan. 2025")
