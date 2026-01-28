# pages/1_Global_Explanation.py

import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from google import genai
from google.genai import types


st.set_page_config(page_title="Model", layout="wide")

# SIDEBAR SETUP
with st.sidebar:
    for _ in range(22):
        st.sidebar.write("")
    st.divider()    
    dev_mode = st.toggle("Developer Mode", value=True, 
                         help="Disables AI API calls to save daily quota during UI design.")

# PAGE LOGIC

if 'explainer' not in st.session_state:
    st.warning("Please go to the **Introduction** page first to train the model!")
else:
    # 1. Retrieve data from session state
    explainer = st.session_state['explainer']
    pipeline = st.session_state['pipeline']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    feature_names = st.session_state['feature_names']
    preprocessor = pipeline.named_steps['preprocessor']

    # 2. Initialize AI narrative storage
    if "ai_narrative" not in st.session_state:
        st.session_state.ai_narrative = None

    # 3. Title and description
    st.markdown("# Model Insights")
    st.write("High-level view of performance metrics and model insights")
    st.write("") 


    # 4. TABS FOR CONSOLIDATED VIEW
    tab_accuracy, tab_features = st.tabs(["Model Accuracy", "Features"])

    # 5. Tab Model Accuracy
    with tab_accuracy:

        # Custom CSS for a borderless, typographic layout (Gemini)
        st.markdown("""
            <style>
            .matrix-wrapper {
                margin-top: 20px;
            }
            .matrix-label {
                font-size: 0.9rem;
                color: #636e72;
                margin-bottom: 0px;
                line-height: 1.2;
            }
            .matrix-value {
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 20px;
            }
            .risk-text { color: #d63031; } /* Red for Default Risk */
            .leakage-text { color: #e67e22; } /* Orange for Revenue Leakage */
            .neutral-text { color: #2d3436; }
            
            .accuracy-text {
                font-size: 2.2rem;
                font-weight: 700;
                color: #2d3436;
                text-align: right;
            }
            .accuracy-sub {
                font-size: 0.9rem;
                color: #636e72;
                text-align: right;
                margin-top: -15px;
            }
            </style>
        """, unsafe_allow_html=True)




        cm_col, gap, acc_col = st.columns([1.5, 2, 0.5])

        
        # Model Confussion Matrix
        with cm_col:
            y_pred = pipeline.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # 2x2 typographic grid (Gemini)
            row1_left, row1_right = st.columns(2)
            row2_left, row2_right = st.columns(2)

            with row1_left:
                st.markdown(f'<p class="matrix-label">Approved</p><p class="matrix-value neutral-text">{tp}</p>', unsafe_allow_html=True)
            
            with row1_right:
                st.markdown(f'<p class="matrix-label">Unsafe Approval<br><b>(Default Risk)</b></p><p class="matrix-value risk-text">{fp}</p>', unsafe_allow_html=True)

            with row2_left:
                st.markdown(f'<p class="matrix-label">Unsafe Rejection<br><b>(Revenue Leakage)</b></p><p class="matrix-value leakage-text">{fn}</p>', unsafe_allow_html=True)
            
            with row2_right:
                st.markdown(f'<p class="matrix-label">Rejected</p><p class="matrix-value neutral-text">{tn}</p>', unsafe_allow_html=True)


#            # Convert to a DataFrame with professional labels
#            cm_df = pd.DataFrame(
#                cm, 
#                index=["Actual: REJECT", "Actual: APPROVE"],
#                columns=["Pred: REJECT", "Pred: APPROVE"]
#            )
#
#            # Display as a styled dataframe (Heatmap style)
#            st.dataframe(
#                cm_df.style.background_gradient(cmap='Greens'),
#                use_container_width=True
#            )

        # Model accuracy metric
        with acc_col:
            accuracy = pipeline.score(X_test, y_test)
            st.metric("**Model Accuracy** \n\n**(Test Set)**\n\n", f"{accuracy:.2%}")

        # Confussion Matrix Explanation
        st.write("")
        st.info(f"""
        **Performance Breakdown**
        * From **{len(y_test)}** applications tested, correct was **{tp + tn} ({accuracy: .2%})** application, incorrect - **{len(y_test)-tp - tn}**.
        * **Default Risk:** **{fp}** high-risk borrowers were approved despite a potential default. 
        * **Opportunity Loss:** **{fn}** creditworthy borrowers were rejected, resulting in revenue leakage and loss of market share.
        """)
        st.write("")

        st.divider()
    
    # 6. Tab Features
    with tab_features:
        # SHAP calculations
        if 'shap_values_global' not in st.session_state:
            with st.spinner("Calculating Global Feature Importance..."):
                global_subset_raw = X_test.sample(min(100, len(X_test)), random_state=42)
                global_subset_transformed = preprocessor.transform(global_subset_raw)
                shap_values = explainer.shap_values(global_subset_transformed)

                st.session_state['shap_values_global'] = shap_values
                st.session_state['global_subset_transformed'] = global_subset_transformed

        shap_values = st.session_state['shap_values_global']
        subset_transformed = st.session_state['global_subset_transformed']


        # SHAP plots side-by-side
        fiportance_col, gap, fimpact_col = st.columns([2, 0.5, 2])

        with fiportance_col:
            st.subheader("Feature Importance")
            st.markdown("Shows which features mattered most on average")

            fig1, ax1 = plt.subplots() 
            shap.summary_plot(
                shap_values, 
                subset_transformed, 
                plot_type="bar", 
                show=False, 
                feature_names=feature_names,
                max_display=10
            )
            st.pyplot(fig1, use_container_width=True)

        with fimpact_col:
            st.subheader("Feature Impact")
            st.markdown("Shows how high/low values affected the outcome")

            fig2, ax2 = plt.subplots()
            shap.summary_plot(
                shap_values, 
                subset_transformed, 
                show=False, 
                feature_names=feature_names,
                max_display=10
            )
            st.pyplot(fig2, use_container_width=True)

        st.divider()

        # Contextual explanation with LLM (Gemini)
        # Converting SHAP values into a format the AI can read
        def get_model_context(feature_names, shap_values, X_transformed):
            importances = np.abs(shap_values).mean(0)
            indices = np.argsort(importances)[::-1]

            summary = "Global Model Logic (Feature Impact & Direction):\n"
            for i in range(min(5, len(indices))):
                idx = indices[i]
                name = feature_names[idx]
                # Calculate correlation to see if high feature values lead to Approval or Rejection
                # This explains the 'Beeswarm' color and position logic
                feature_vals = X_transformed[:, idx]
                shap_vals = shap_values[:, idx]
                correlation = np.corrcoef(feature_vals, shap_vals)[0, 1]

                direction = "High values = Approval" if correlation > 0 else "High values = Rejection"
                summary += f"- {name}: Impact {importances[idx]:.4f} | {direction}\n"
             
            return summary

        # Configure the AI Explainer
        st.subheader("AI Explanation: Executive Summary")
        
        # Initialize session state for the narrative
        if "ai_narrative" not in st.session_state:
            st.session_state.ai_narrative = None
        
        if dev_mode:
            st.info("**Safe Mode Active**: API calls are paused to protect quota.")
            if st.session_state.ai_narrative is None:
                st.session_state.ai_narrative = "(AI Narrative placeholder for UI layout testing)"
        else:
            if st.session_state.ai_narrative is None or st.session_state.ai_narrative.startswith("*("):
                try:
                    with st.spinner("Generating AI Analysis..."):
                        client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
                        # Use your helper function to get context
                        snapshot = get_model_context(feature_names, shap_values, subset_transformed)

                        config = types.GenerateContentConfig(
                            #max_output_tokens=120,
                            temperature=0.1,
                            system_instruction="""You are a Senior Risk Auditor. 
                            1. Summarize the lending strategy based on the provided scores.
                            2. Explain not just what features matter, but HOW their values (high/low) influence the decision based on the provided directions.
                            3. Use professional language."""
                        )
        
                        response = client.models.generate_content(
                            model='gemini-3-flash-preview', 
                            contents=f"{snapshot}\n\nExplain the strategic logic of these top features.", 
                            config=config
                        )
                        st.session_state.ai_narrative = response.text
                except Exception as e:
                    st.error(f"Quota/API Error: {e}")
        
        # Display the result
        if st.session_state.ai_narrative:
            st.write(st.session_state.ai_narrative)

            st.caption("✨ Generated by Gemini. Gemini can make mistakes, including about people, so double-check it. [Your privacy and Gemini](https://support.google.com/gemini?p=privacy_notice)")

            col_fb1, col_fb2, _ = st.columns([0.1, 0.1, 2])
            with col_fb1:
                if st.button("👍"): st.toast("Thanks for the feedback!")
            with col_fb2:
                if st.button("👎"): st.toast("Feedback recorded for re-training.")
    
            # Only show refresh if we are NOT in dev mode
            if not dev_mode:
                if st.button("🔄 Refresh Analysis"):
                    st.session_state.ai_narrative = None
                    st.rerun()
