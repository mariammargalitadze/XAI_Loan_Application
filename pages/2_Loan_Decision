import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import dice_ml
from openai import OpenAI
from google import genai
from google.genai import types


st.set_page_config(page_title="Loan Decision", layout="wide")

# SIDEBAR SETUP
with st.sidebar:
    
    # 1. Applicant Selection
    if 'X_test' in st.session_state:
        selected_idx = st.selectbox(
            "Select Loan Applicant",
            options=range(len(st.session_state['X_test'])),
            index=st.session_state.get('selected_index', 0)
        )
        st.session_state['selected_index'] = selected_idx
    
    # 2. Formatting Spacer (Pushes content to bottom)
    for _ in range(17): 
        st.write("") 
    st.divider()
    dev_mode = st.toggle("Developer Mode", value=True, help="Disables API to save quota.")
    
    # 3. Reset AI Narrative if switching to Live Mode
    if not dev_mode and (st.session_state.get('local_narrative') or "").startswith("*("):
        st.session_state.local_narrative = None


# MAIN DATA RETRIEVAL
if 'pipeline' not in st.session_state:
    st.warning("Please go to the **Introduction** page first to train the model!")
else:
    # Retrieve data
    explainer = st.session_state['explainer']
    pipeline = st.session_state['pipeline']
    X_test = st.session_state['X_test']
    feature_names = st.session_state['feature_names']
    preprocessor = pipeline.named_steps['preprocessor']

    # Reset AI Logic on applicant change
    if 'last_selected_idx' not in st.session_state:
        st.session_state.last_selected_idx = selected_idx
    
    if st.session_state.last_selected_idx != selected_idx:
        st.session_state.local_narrative = None
        st.session_state.last_selected_idx = selected_idx

    if 'local_narrative' not in st.session_state:
        st.session_state.local_narrative = None

    # 4. Instance
    instance_raw = X_test.iloc[[selected_idx]]
    instance_transformed = preprocessor.transform(instance_raw)
    prob = pipeline.predict_proba(instance_raw)[0][1]
    
    # 5. Title and description
    st.title(f"Loan Decision - Loan Applicant N - {selected_idx}")

    status = "✅ Approved" if prob > 0.5 else "❌ Rejected"
    st.write(f"**{status} - {prob:.1%}**")


    # 6. Tabs for consolidated view
    tab_waterfall, tab_neighbors, tab_counterfactual, tab_perturbation, tab_decision = st.tabs([
        "📊 Why was this decided?",
        "👥 Who is similar?",
        "🛠️ What could change?",
        "🔄 What influences most?",
        "✅ Final Decision"
    ])

    
    # 7. Tab Waterfall
    with tab_waterfall:
        # --- SHARED DATA PREP (Calculating Percentage Points) ---
        try:
            if 'classifier' in pipeline.named_steps:
                model = pipeline.named_steps['classifier']
            else:
                model = pipeline.steps[-1][1]
            
            background = preprocessor.transform(X_test.sample(min(50, len(X_test)), random_state=42))
            prob_explainer = shap.TreeExplainer(model, data=background, model_output="probability")
            shap_values_prob = prob_explainer(instance_transformed)
            
            # Extract % Point values
            vals_pp = shap_values_prob.values[0, :, 1] * 100 if len(shap_values_prob.values.shape) > 2 else shap_values_prob.values[0] * 100
            base_pp = shap_values_prob.base_values[0, 1] * 100 if len(shap_values_prob.base_values.shape) > 1 else shap_values_prob.base_values[0] * 100
            final_pp = base_pp + np.sum(vals_pp)

            # CUSTOM LABELS: Name first, then Value rounded to 2 digits
            row_values = instance_transformed[0]
            if hasattr(row_values, "toarray"): 
                row_values = row_values.toarray()[0]
            
            # FeatureName = 0.00 format for the plot axis
            custom_labels = [f"{name} = {val:.2f}" for name, val in zip(feature_names, row_values)]

            exp_obj = shap.Explanation(
                values=vals_pp, 
                base_values=base_pp, 
                feature_names=custom_labels
            )

            # Plot + Summary
            col_wf, col_summary = st.columns([2, 1.2])
            
            with col_wf:
                st.subheader("Feature Contribution (% Points)")
                fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
                # Plotting using our updated labels
                shap.plots.waterfall(exp_obj, show=False)
                plt.xlabel("Impact on Approval Probability")
                st.pyplot(fig_wf)
                plt.close(fig_wf)

            with col_summary:
                st.subheader("Quick Summary")
                st.markdown(f"""
                - **Starting Point (Average):** {base_pp:.1f}%
                - **Final Score:** {final_pp:.1f}%
                
                **How to read:**  
                Red bars push the score **UP** (Approval).  
                Blue bars pull the score **DOWN** (Risk).
                """)
                st.divider()
                st.caption("The numbers represent how many 'Percentage Points' each factor added or removed from the average.")

        
            
            st.markdown("---")
            st.subheader("Ask Gemini about this Plot")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display Chat History with Feedback Buttons
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Add feedback buttons ONLY for assistant messages
                    if message["role"] == "assistant":
                        # Use small columns to keep icons tight together
                        btn_col1, btn_col2, _ = st.columns([0.05, 0.05, 0.9])
                        with btn_col1:
                            if st.button("👍", key=f"like_{i}"):
                                st.toast(f"Thanks! Feedback saved for response #{i//2 + 1}", icon="✅")
                        with btn_col2:
                            if st.button("👎", key=f"dislike_{i}"):
                                st.toast("Feedback recorded. We'll work to improve.", icon="📝")

            # Chat Input
            if user_query := st.chat_input("Ex: Why is the Loan Duration bar so big?"):
                # Add user message to state
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.rerun() # Refresh to show user message immediately

            # Handle generating the response if the last message was from the user
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                with st.chat_message("assistant"):
                    if dev_mode:
                        response_text = "*(Developer Mode: AI Chat response is a placeholder)*"
                    else:
                        try:
                            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                            
                            # Top 3 drivers for context
                            top_3 = np.argsort(np.abs(vals_pp))[::-1][:3]
                            context_data = "\n".join([f"- {feature_names[i]}: {vals_pp[i]:+.1f} points" for i in top_3])

                            chat_prompt = f"""
                            You are a Credit Risk Mentor.
                            
                            PLOT DATA:
                            - Average Start: {base_pp:.1f}%
                            - Final Probability: {final_pp:.1f}%
                            - Top 3 Drivers: {context_data}
                            
                            USER QUESTION: {st.session_state.chat_history[-1]["content"]}
                            
                            INSTRUCTIONS:
                            Explain using 'Percentage Points'. Be professional and brief.
                            """
                            
                            response = client.models.generate_content(model="gemini-3-flash-preview", contents=chat_prompt)
                            response_text = response.text
                        except Exception as e:
                            response_text = f"Sorry, I ran into an error: {e}"
                    
                    # Add assistant response to state and refresh to show buttons
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Waterfall/Chat Error: {e}")

            
    
    
    
#    # 7. Tab Waterfall
#    with tab_waterfall:
#        col_wf, gap, col_ai = st.columns([2, 0.1, 1.5])
#        
#        with col_wf:
#            st.subheader("Feature Contribution (Percentage Points)")
#            
#            try:
#                # A. Extract Model & Create Background
#                if 'classifier' in pipeline.named_steps:
#                    model = pipeline.named_steps['classifier']
#                else:
#                    model = pipeline.steps[-1][1]
#                
#                background = preprocessor.transform(X_test.sample(min(50, len(X_test)), random_state=42))
#
#                # B. Initialize Probability Explainer
#                prob_explainer = shap.TreeExplainer(
#                    model, 
#                    data=background, 
#                    feature_perturbation="interventional",
#                    model_output="probability"
#                )
#
#                # C. Calculate Values
#                shap_values_prob = prob_explainer(instance_transformed)
#
#                # D. Extract Class 1 (Approval) Values
#                if hasattr(shap_values_prob, "values"):
#                    vals = shap_values_prob.values
#                    base = shap_values_prob.base_values
#                else:
#                    vals = shap_values_prob
#                    base = prob_explainer.expected_value
#
#                if len(vals.shape) > 1 and vals.shape[-1] == 2:
#                    vals = vals[0, :, 1]
#                    base = base[0, 1]
#                elif len(vals.shape) > 1:
#                     vals = vals[0]
#                     base = base[0] if isinstance(base, (list, np.ndarray)) else base
#
#                # E. PERCENTAGE POINTS
#                vals_percent = vals * 100
#                base_percent = base * 100
#
#                # F. CUSTOM LABELS (Name = 0.00)
#                row_values = instance_transformed[0]
#                if hasattr(row_values, "toarray"):
#                    row_values = row_values.toarray()[0]
#                
#                custom_labels = [f"{name} = {val:.2f}" for name, val in zip(feature_names, row_values)]
#
#                # Explanation Object
#                exp = shap.Explanation(
#                    values=vals_percent, 
#                    base_values=base_percent, 
#                    data=None, 
#                    feature_names=custom_labels
#                )
#
#                # G. Plot
#                fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
#                shap.plots.waterfall(exp, show=False)
#                
#
#                plt.xlabel("Contribution to Approval Probability (Percentage Points)")
#                plt.tight_layout()
#                st.pyplot(fig_wf)
#                plt.close(fig_wf)
#
#                # H. Note on Terminology
#                st.info("""
#                **📝 Note on Terminology:**
#                * **$E[f(x)]$ - Average Approval Rate:** The starting baseline. It represents the average approval probability for the entire dataset.
#                * **$f(x)$ - Applicant Probability:** The final score for this specific applicant. It is the result of adding all the +/- feature impacts to the baseline.
#                """)
#
#            except Exception as e:
#                st.warning(f"Could not calculate percentage points. Showing raw values. Error: {e}")
#                shap_val = explainer.shap_values(instance_transformed)
#                shap_val_to_plot = shap_val[0] if len(shap_val.shape) > 1 else shap_val
#                
#                exp = shap.Explanation(
#                    values=shap_val_to_plot, 
#                    base_values=explainer.expected_value, 
#                    data=instance_transformed[0], 
#                    feature_names=feature_names
#                )
#                
#                fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
#                shap.plots.waterfall(exp, show=False)
#                st.pyplot(fig_wf)
#                plt.close(fig_wf)
#
#            except Exception as e:
#                st.warning(f"Could not calculate percentage points. Showing raw values. Error: {e}")
#                # Fallback logic here (omitted for brevity, keep your existing fallback if needed)
#
#
#    # Right Column: AI Contextual Explanation (GEMINI)
#    with col_ai:
#        st.subheader("Local AI Executive Summary")
#        
#        def get_applicant_story(feature_names, shap_values, instance_raw, baseline_value, final_prob):
#            preprocessor = st.session_state['pipeline'].named_steps['preprocessor']
#            instance_transformed = preprocessor.transform(instance_raw)
#            if hasattr(instance_transformed, "toarray"):
#                instance_transformed = instance_transformed.toarray()
#            
#            def sigmoid(x): return 1/(1+np.exp(-x))
#            total_log_odds = baseline_value + np.sum(shap_values)
#
#            indices = np.argsort(np.abs(shap_values))[::-1] 
#            positive_features = []
#            negative_features = []
#            
#            for i in indices:
#                # Calculate contribution in probability space for the text
#                prob_without = sigmoid(total_log_odds - shap_values[i])
#                prob_delta = sigmoid(total_log_odds) - prob_without
#                val = instance_transformed[0, i]
#                
#                # Formatted string: Name first, then Value (2 decimals)
#                feature_info = f"{feature_names[i]} = {val:.2f} (Impact: {prob_delta:+.1%})"
#
#                if shap_values[i] > 0 and len(positive_features) < 3:
#                    positive_features.append(feature_info)
#                elif shap_values[i] <0 and len(negative_features) < 3:
#                    negative_features.append(feature_info)
#            
#            return {
#                'baseline': f'{baseline_value: .1%}' if -1 <= baseline_value <= 1 else "Log-Odds Baseline",
#                'final': f'{final_prob: .1%}',
#                'positives': "\n".join(positive_features),
#                'negatives': "\n".join(negative_features)
#            }
#
#        if dev_mode:
#            if st.session_state.local_narrative is None:
#                st.session_state.local_narrative = "(AI Narrative placeholder for UI layout testing)"
#            st.info("Safe Mode Active: API calls are paused to protect quota.")
#        else:
#            if st.session_state.local_narrative is None or st.session_state.local_narrative.startswith("*("):
#                try:
#                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
#                    
#                    # Use standard explainer values for AI delta calc
#                    shap_val = explainer.shap_values(instance_transformed)
#                    shap_val_for_ai = shap_val[0] if len(shap_val.shape) > 1 else shap_val
#                    
#                    story_data = get_applicant_story(feature_names, shap_val_for_ai, instance_raw, explainer.expected_value, prob)
#
#                    prompt = f"""
#                    TASK: 
#                    1. Provide a final auditor's judgment on creditworthiness. 
#                    2. Be consize.
#                    """
#                    
#                    config = types.GenerateContentConfig(
#                        temperature=0.2,
#                        system_instruction="You are a Lead Credit Risk Auditor. Write a 3-paragraph summary. Use professional banking terminology."
#                    )
#
#                    response = client.models.generate_content(
#                        model="gemini-3-flash-preview", 
#                        contents=prompt,
#                        config=config
#                    )
#                    st.session_state.local_narrative = response.text
#                except Exception as e:
#                    st.error(f"Narrative Generation Error: {e}")
#
#        # Display Narrative
#        if st.session_state.local_narrative:
#            st.markdown(
#                f"""
#                <div style="padding: 15px; border: 1px solid #e6e9ef; border-radius: 5px; background-color: #f9f9f9; color: #333; font-family: sans-serif; line-height: 1.6;">
#                    {st.session_state.local_narrative}
#                </div>
#                """,
#                unsafe_allow_html=True
#            )
#            
#            st.caption("✨ Generated by Gemini. Gemini can make mistakes, including about people, so double-check it. [Your privacy and Gemini](https://support.google.com/gemini?p=privacy_notice)")
#            
#            fb_col1, fb_col2, _ = st.columns([0.15, 0.15, 0.7])
#            with fb_col1:
#                if st.button("👍", key="like"): st.toast("Thanks for the feedback!")
#            with fb_col2:
#                if st.button("👎", key="dislike"): st.toast("Feedback recorded.")
#                
                

    # 8. Tab Neighbors
    with tab_neighbors:
        col_list, gap, col_plot = st.columns([2, 0.5, 2])

        # A. Similar Cases
        X_test_transformed = preprocessor.transform(X_test)
        # Find 6 neighbors (the 1st one is the applicant themselves)
        nn = NearestNeighbors(n_neighbors=4)
        nn.fit(X_test_transformed)
        distances, indices = nn.kneighbors(instance_transformed)

        # Get raw data and predictions for the neighbors
        similar_indices = indices[0]
        similar_cases_raw = X_test.iloc[similar_indices].copy()
        
        # Calculate probabilities for the neighbors to add to the table
        neighbor_probs = pipeline.predict_proba(similar_cases_raw)
        probs_formatted = [f"{p[1]:.1%}" for p in neighbor_probs]

        
        

        with col_list:
            st.subheader("Feature Benchmarking")
            
            # 1. PREPARE DATA
            # Prepare Transposed DataFrame with clean IDs
            bench_df = similar_cases_raw.T
            # Column 0 is CURRENT, 1-3 are neighbors
            column_names = ["CURRENT"] + [f"N{i}" for i in range(1, 4)]
            bench_df.columns = column_names
            
            
            # DECODER: Map coded values to readable text
            
            decoder_map = {
                'y': 'Yes', 'b': 'Medium', 'w': 'Waiting', 'i': 'Increasing',
                'f': 'False', 'u': 'No', 'g': 'Guarantor', 'p': 'Parent',
                'l': 'Long-Term', 't': 'True', 'v': 'Medium', 'h': 'Housing',
                'bb': 'Bank Balance', 'ff': 'Free/Clear', 'j': 'Junior', 'z': 'Zero/None',
                'dd': 'Direct Debit', 'n': 'Negative', 'o': 'Other'
            }

            def decode_dataframe(df):
                """Helper to apply the map to an entire dataframe"""
                return df.replace(decoder_map)

            bench_df_readable = decode_dataframe(bench_df)

            # Add a "RESULT" row at the top
            result_row = pd.DataFrame([probs_formatted], columns=column_names, index=["Approval Prob"])
            final_bench = pd.concat([result_row, bench_df_readable])

           
            # [NEW] VISUAL VIEW OPTIONS (Filter & Transpose) (GEMINI)
            with st.expander(":gray[🌪️ View Options (Filter & Transpose)]", expanded=False):
                opt_col1, opt_col2 = st.columns([1, 2])
                
                with opt_col1:
                    # A. ROW vs COLUMN Toggle
                    view_mode = st.radio(
                        "Display Orientation:", 
                        ["Features as Rows", "Features as Columns"], 
                        horizontal=True
                    )
                
                with opt_col2:
                    # B. FILTER Selection
                    # We grab the index (Feature Names) to populate the filter list
                    available_features = final_bench.index.tolist()
                    selected_features = st.multiselect(
                        "Show specific attributes:", 
                        options=available_features,
                        default=available_features, # Default to showing all
                        placeholder="Select features to compare..."
                    )

        
            # APPLY VISUALS (Local Display Logic)
            # 1. Filter Rows based on selection
            if selected_features:
                view_df = final_bench.loc[selected_features]
            else:
                view_df = final_bench # Show all if nothing selected

            # 2. Transpose if user selected "Features as Columns"
            if view_mode == "Features as Columns":
                view_df = view_df.T
                highlight_axis = 0 # Highlight vertical max if transposed
            else:
                highlight_axis = 1 # Highlight horizontal max (standard)

            # 3. Render Final Table
            st.dataframe(
                view_df.style.highlight_max(axis=highlight_axis, color='#e7f4f9') 
                .format(precision=2),
                use_container_width=True
            )
            
            st.caption("🔍 Compare the 'CURRENT' values against 'Neighbor' columns to find the tipping points.")



        
        
#        with col_list:
#            st.subheader("Feature Benchmarking")
#            
#            # Prepare Transposed DataFrame with clean IDs
#            bench_df = similar_cases_raw.T
#            # Column 0 is CURRENT, 1-3 are neighbors
#            column_names = ["CURRENT"] + [f"N{i}" for i in range(1, 4)]
#            bench_df.columns = column_names
#            
#            
#
#            
#            # DECODER: Map coded values to readable text
#            decoder_map = {
#                # Example mappings (adjust these to match your specific dataset)
#                'y': 'Yes',
#                'b': 'Business',
#                'w': 'Waiting',
#                'i': 'Icreasing',
#                'f': 'False',
#                'u': 'No',
#                'g': 'Guarantor',
#                'p': 'Parent',
#                'l': 'Long-Term',
#                't': 'Temporary',
#                'v': 'Vehicle',
#                'h': 'Housing',
#                'bb': 'Bank Balance',
#                'ff': 'Free/Clear',
#                'j': 'Junior',
#                'z': 'Zero/None',
#                'dd': 'Direct Debit',
#                'n': 'Negative',
#                'o': 'Other'
#            }
#
#            def decode_dataframe(df):
#                """Helper to apply the map to an entire dataframe"""
#                return df.replace(decoder_map)
#
#            bench_df_readable = decode_dataframe(bench_df)
#
#            # Add a "RESULT" row at the top so you see why they are similar
#            result_row = pd.DataFrame([probs_formatted], columns=column_names, index=["Approval Prob"])
#            bench_df = pd.concat([result_row, bench_df_readable]) #bench_df
#            
#            st.dataframe(bench_df.style.highlight_max(axis=1, color='#e7f4f9'), use_container_width=True)
#            st.caption("🔍 Compare the 'CURRENT' values against 'Neighbor' columns to find the tipping points.")
#






        with col_plot:
            st.subheader("Similar Decisions")

            # PCA Setup
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_test_transformed)
            current_pca = pca.transform(instance_transformed)
            neighbors_pca = pca.transform(X_test_transformed[similar_indices[1:]])

            # Decision Boundary "Zones"
            padding = 1.5
            x_min, x_max = X_pca[:, 0].min() - padding, X_pca[:, 0].max() + padding
            y_min, y_max = X_pca[:, 1].min() - padding, X_pca[:, 1].max() + padding
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
            
            # Map grid back to 37-feature space and predict
            grid_points_transformed = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
            Z = pipeline[-1].predict_proba(grid_points_transformed)[:, 1]
            Z = Z.reshape(xx.shape)

            # PLOTTING
            fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
            
            # Draw Probability Zones (Green = Safe, Red = Risk)
            contour = ax_pca.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.3)
            # Draw the 0.5 Decision Boundary line
            ax_pca.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
            
            # Background Population (Grey dots)
            ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c='grey', alpha=0.1, s=10, label='Population')
            
            # Plot and LABEL each neighbor (Connects visual to table)
            for i, (x, y) in enumerate(neighbors_pca):
                label = f"N{i+1}"
                ax_pca.scatter(x, y, c='blue', s=100, edgecolors='white', zorder=5)
                ax_pca.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                                fontsize=9, fontweight='bold', color='blue')

            # Plot Target (Current Applicant)
            ax_pca.scatter(current_pca[0, 0], current_pca[0, 1], c='red', marker='*', s=450, 
                           label='TARGET', edgecolors='black', zorder=10)
            
            ax_pca.set_xlabel("PC1 (Variance Driver 1)")
            ax_pca.set_ylabel("PC2 (Variance Driver 2)")
            ax_pca.legend(loc='upper right')
            st.pyplot(fig_pca)
            plt.close(fig_pca)
            
            # Explanation
            st.info("""
            **How to interpret this map:**
            - **Dashed Line:** This is the model's 'Point of No Return'. Points to the left/right represent the jump between Approved and Rejected.
            - **Zones:** Green areas show high confidence in approval; Red shows high risk. 
            - **Neighbor Labels:** These IDs correspond exactly to the columns in the table on the left.
            - **Distance:** If the Red Star is far from a Blue Neighbor, look at the table to see which feature (e.g., Income) has the largest difference.
            """)


##        Without a decision boundary, visualizing the decision distance is practically impossible, so this option was declined. 
##        with col_plot:
##            st.subheader("Decision Space Visualization")
##            
##            pca = PCA(n_components=2)
##            X_pca = pca.fit_transform(X_test_transformed)
##            current_pca = pca.transform(instance_transformed)
##            neighbors_pca = pca.transform(X_test_transformed[similar_indices[1:]])
##
##            fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
##            
##            # 1. Background Population
##            ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c='grey', alpha=0.1, s=10)
##
##            # 2. Plot and LABEL each neighbor
##            for i, (x, y) in enumerate(neighbors_pca):
##                neighbor_label = f"Neighbor {i+1}"
##                ax_pca.scatter(x, y, c='blue', s=100, edgecolors='white', zorder=5)
##                # This line connects the dot to the table!
##                ax_pca.annotate(neighbor_label, (x, y), xytext=(5, 5), 
##                                textcoords='offset points', fontsize=9, fontweight='bold')
##
##            # 3. Target
##            ax_pca.scatter(current_pca[0, 0], current_pca[0, 1], c='red', marker='*', s=400, 
##                           label='TARGET (Current)', edgecolors='black', zorder=10)
##            
##            ax_pca.set_xlabel("PC1")
##            ax_pca.set_ylabel("PC2")
##            st.pyplot(fig_pca)
##            plt.close(fig_pca)


    # 9. Tab Counterfactual
    if 'tab_counterfactual' not in locals():
        # Creating a tab
        tab_counterfactual = st.container()

    with tab_counterfactual:
        st.subheader("Alternative Paths")
        st.write("This table shows the minimum changes required to flip the decision. Unchanged values are hidden for clarity.")

        
        
        
        # Check for data availability
        if 'train_df' not in st.session_state:
            st.error("Missing training data. Please go to the Introduction page and reload the dataset.")
        else:
            # Load objects from session state
            train_df = st.session_state['train_df']
            
            # FILTER OPTIONS (Using Primary Columns)
            with st.expander("🌪️ Feature Constraints (Visual Prototype)", expanded=False):
                st.caption("Select features to lock or prioritize for the alternative paths:")
                
                # 1. Get Primary Columns (Exclude the Target 'Approved')
                # This gives the original 15 columns
                primary_features = [c for c in train_df.columns if c != 'Approved']
                
                # 2. Create Grid
                f_cols = st.columns(4)
                for i, feature in enumerate(primary_features):
                    # Distribute checkboxes across 4 columns
                    f_cols[i % 4].checkbox(feature, value=False, key=f"cf_viz_filter_{i}")
                
                st.button("Apply Constraints (Demo)", help="This is a visual element only.")




        if 'train_df' not in st.session_state:
            st.error("Missing training data. Please go to the Introduction page and reload the dataset.")
        else:
            # Load objects from session state
            train_df = st.session_state['train_df']
            continuous_features = [c for c in st.session_state.get('continuous_cols', []) if c in train_df.columns]
            expected_features = X_test.columns.tolist()

            try:
                # A. SETUP DICE
                d = dice_ml.Data(dataframe=train_df, continuous_features=continuous_features, outcome_name='Approved')
                m = dice_ml.Model(model=pipeline, backend="sklearn")
                exp_dice = dice_ml.Dice(d, m, method="random")

                with st.spinner("Generating 3 Counterfactual Scenarios..."):
                    # B. GENERATE 3 COUNTERFACTUALS
                    query_instance = X_test.iloc[[selected_idx]]
                    dice_exp = exp_dice.generate_counterfactuals(
                        query_instance, 
                        total_CFs=3, 
                        desired_class="opposite"
                    )
                    cf_df = dice_exp.cf_examples_list[0].final_cfs_df

                    # C. POSITIONING: Move 'Approved' to first column
                    cols = ['Approved'] + [c for c in cf_df.columns if c != 'Approved']
                    cf_df = cf_df[cols]

                    # Prepare "Original Input" row
                    original_input = instance_raw.copy()
                    original_input['Approved'] = 0 if prob <= 0.5 else 1
                    original_input = original_input[cols] 
                    original_input.index = ['Original Input']

                    # D. MASKING: Replace unchanged features with "—"
                    display_cf = cf_df.copy()
                    for i in range(len(display_cf)):
                        for col in display_cf.columns:
                            if col == 'Approved':
                                continue
                            # Compare CF value to the Original row
                            if str(display_cf.iloc[i][col]) == str(original_input.iloc[0][col]):
                                display_cf.iloc[i, display_cf.columns.get_loc(col)] = "—"

                    display_cf.index = [f"Counterfactual {i+1}" for i in range(len(display_cf))]
                    final_table = pd.concat([original_input, display_cf], axis=0)

                    # E. STYLING: No internal grid lines, thick separation line
                    def style_audit_table(df):
                        return df.style.set_table_styles([
                            # Remove all internal grid borders
                            {'selector': 'th, td', 'props': [('border', 'none !important')]},
                            # Thin line under the header
                            {'selector': 'thead th', 'props': [('border-bottom', '1px solid #ccc !important')]},
                            # THICK VIVID LINE only under the first row (Original Input)
                            {'selector': 'tbody tr:nth-child(1) td', 
                             'props': [('border-bottom', '3px solid black !important')]},
                            # General padding and alignment
                            {'selector': 'td', 'props': [('text-align', 'center'), ('padding', '8px')]}
                        ]).map(lambda v: 'color: #28a745; font-weight: bold;' if v == 1 else ('color: #dc3545;' if v == 0 else ''), subset=['Approved']) \
                          .apply(lambda x: ['background-color: #f8f9fb; font-weight: bold;' if x.name == 'Original Input' else '' for _ in x], axis=1)

                    # Render the table
                    st.table(style_audit_table(final_table))

                # F. LOCAL PCA VISUALIZATION
                st.markdown("---")
                col_text, col_plot = st.columns([1, 2])

                with col_text:
                    st.subheader("Counterfactual Decisions")
                    st.markdown(f"""
                    - **Red Star**: Current Applicant ({'Rejected' if prob <= 0.5 else 'Approved'})
                    - **Grey Dots**: The 3 counterfactual paths shown in the table.
                    - **Dashed Line**: The 'Decision Boundary' where the model flips its mind.
                    """)

                with col_plot:
                    # 1. Local PCA Fit
                    X_test_transformed = preprocessor.transform(X_test)
                    pca_local = PCA(n_components=2)
                    X_pca = pca_local.fit_transform(X_test_transformed)

                    # 2. Project Points
                    current_pca_pt = pca_local.transform(instance_transformed)
                    # Filter CF features to match training for transformation
                    cf_raw_only = cf_df[expected_features].copy().astype(X_test.dtypes)
                    cf_transformed_all = preprocessor.transform(cf_raw_only)
                    cf_pca_pts = pca_local.transform(cf_transformed_all)

                    # 3. Decision Zones (Meshgrid)
                    padding = 2.0
                    x_min, x_max = X_pca[:, 0].min() - padding, X_pca[:, 0].max() + padding
                    y_min, y_max = X_pca[:, 1].min() - padding, X_pca[:, 1].max() + padding
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

                    grid_points_encoded = pca_local.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
                    classifier_model = pipeline.steps[-1][1] # Direct model access
                    Z_local = classifier_model.predict_proba(grid_points_encoded)[:, 1].reshape(xx.shape)

                    # 4. Plot
                    fig_cf, ax_cf = plt.subplots(figsize=(7, 4.5))
                    ax_cf.contourf(xx, yy, Z_local, levels=20, cmap='RdYlGn', alpha=0.25)
                    ax_cf.contour(xx, yy, Z_local, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')

                    # Plot Points
                    ax_cf.scatter(current_pca_pt[0, 0], current_pca_pt[0, 1], c='red', marker='*', s=350, label='Current', zorder=5)
                    ax_cf.scatter(cf_pca_pts[:, 0], cf_pca_pts[:, 1], c='grey', marker='o', s=120, label='Goals', zorder=5)

                    # Arrows
                    for i in range(len(cf_pca_pts)):
                        ax_cf.annotate("", xy=(cf_pca_pts[i, 0], cf_pca_pts[i, 1]), 
                                       xytext=(current_pca_pt[0, 0], current_pca_pt[0, 1]),
                                       arrowprops=dict(arrowstyle="->", color='black', alpha=0.2))
                        

                        # Labels
                        ax_cf.annotate(
                            text=str(i+1), 
                            xy=(cf_pca_pts[i, 0], cf_pca_pts[i, 1]),
                            xytext=(5, 5), # Shift text 5 points up and right
                            textcoords='offset points',
                            fontsize=9, 
                            fontweight='bold', 
                            color='black'
                        )
                        
                    ax_cf.set_xlabel("Principal Component 1")
                    ax_cf.set_ylabel("Principal Component 2")
                    ax_cf.legend(loc='lower right', fontsize='small')
                    st.pyplot(fig_cf)
                    plt.close(fig_cf)

            except Exception as error:
                st.error(f"Audit Analysis Error: {error}")

    
    # 10. Tab Feature Perturbation
    with tab_perturbation:
            # 1. Get original data for the applicant
            original_data = X_test.iloc[[selected_idx]].copy()
            # 2. Prediction for Original Data
            orig_prob = pipeline.predict_proba(original_data)[0][1]
            # 3. Left (Controls) & Right (Results)
            st.write("")
            col1, gap, col2 = st.columns([2, 0.5, 2])
            with col1:
                st.subheader("Perturb Features")
                st.write(f"Modify applicant **#{selected_idx}**'s data to see how the model's decision shifts.")
                # We create a dictionary to hold our modified values
                modified_values = {}
                # Perturb several features
                modified_values['Active Years'] = st.slider(
                    "Active Years (Business Age)", 
                    0, 50, int(original_data['Active Years'].values[0])
                )
                # Perturbing 'Interest Rate'
                modified_values['Interest Rate'] = st.slider(
                    "Requested Interest Rate (%)", 
                    0.0, 30.0, float(original_data['Interest Rate'].values[0]), step=0.5
                )
                # Perturbing 'Prior Default' (Categorical)
                p_default_options = ['t', 'f'] # 't' for true, 'f' for false as per UCI dataset
                modified_values['Prior Default'] = st.selectbox(
                    "Prior Default Status", 
                    options=p_default_options,
                    index=p_default_options.index(original_data['Prior Default'].values[0])
                )
                # Perturbing 'Loan Maturity'
                modified_values['Loan Maturity'] = st.number_input(
                    "Loan Maturity (Months)", 
                    value=int(original_data['Loan Maturity'].values[0])
                )
            # 4. Perturbed Dataframe
            perturbed_data = original_data.copy()
            for col, val in modified_values.items():
                perturbed_data[col] = val
            # 5. Prediction for Perturbed Data
            new_prob = pipeline.predict_proba(perturbed_data)[0][1]
            delta = new_prob - orig_prob
            with col2:
                st.subheader("Prediction Impact")
                # Metric showing the shift
                st.metric(
                    label="New Approval Probability", 
                    value=f"{new_prob:.2%}", 
                    delta=f"{delta:.2%}"
                )
                # Comparison Chart
                chart_data = pd.DataFrame({
                    "Scenario": ["Original", "Perturbed"],
                    "Probability": [orig_prob, new_prob]
                })
                st.bar_chart(chart_data, x="Scenario", y="Probability")
                # Logic feedback
                if new_prob > 0.5 and orig_prob <= 0.5:
                    st.success("🌟 Outcome Changed: The perturbation turned a REJECTION into an APPROVAL.")
                elif new_prob <= 0.5 and orig_prob > 0.5:
                    st.error("⚠️ Outcome Changed: The perturbation turned an APPROVAL into a REJECTION.")
                else:
                    st.info("""
                    **How to interpret chart:**
                    - This page performs Sensitivity Analysis.
                    - By isolating specific features and changing them while keeping everything else constant. 
                    - We can see the 'marginal effect' of that feature on the final score.
                    - The model outcome status remains the same despite the changes.
                    """)
            st.divider()


    # 11 Tab Decision (Human-in-the-Loop)
    with tab_decision:
        st.subheader("Final Review")
        st.write("Review the model's recommendation and log your final judgment for future model fine-tuning.")

        # Layout: Summary on Left, Action Form on Right
        col_summary, gap, col_action = st.columns([1.5, 0.2, 2])

        # Case
        with col_summary:
            st.caption("Key Applicant Data:")
            # Show a few key raw values for context (adjust columns as needed for your specific dataset)
            # Assuming 'Income', 'Age', etc. might be in your columns. 
            # We display the first 5 columns of the raw instance for quick reference.
            st.dataframe(instance_raw.iloc[:, :].T, use_container_width=True)

        # Human Decision
        with col_action:
            with st.form("human_decision_form"):
                st.subheader("🗳️ Underwriter's Decision")
                
                # 1. The Decision
                human_vote = st.radio(
                    "Final Decision Status:",
                    options=["Approve", "Reject"],
                    horizontal=True,
                    help="Override the model if you believe there is context the AI missed."
                )

                # 2. The Reasoning (Crucial for Fine-Tuning)
                reason_codes = st.multiselect(
                    "Primary Factors for Decision:",
                    options=["Income Stability", "Debt Ratio", "Collateral", "Employment History", "Model Error", "Policy Exception", "Other"],
                    default=None
                )
                
                comments = st.text_area(
                    "Detailed Notes & Reasoning:",
                    placeholder="E.g., Applicant has strong cash reserves not reflected in the debt-to-income ratio...",
                    help="This text will be used to label data for retraining."
                )

                # 3. Submit
                submitted = st.form_submit_button("🔒 Log Final Decision")

            if submitted:
                # 4. Post-submission Comparison
                st.divider()
                st.subheader("⚖️ Decision Comparison")
                
                c1, c2 = st.columns(2)
                
                # Model Side
                with c1:
                    st.markdown(f"**🤖 AI Model**")
                    if prob > 0.5:
                        st.success(f"Approved ({prob:.1%})")
                    else:
                        st.error(f"Rejected ({1-prob:.1%})")
                
                # Human Side
                with c2:
                    st.markdown(f"**👤 Human Auditor**")
                    if human_vote == "Approve":
                        st.success(f"Approved")
                    else:
                        st.error(f"Rejected")

#                # Feedback Storage Simulation
#                # In a real app, this would save to a database (SQL/Snowflake).
#                # Here we save to session state to show it 'worked'.
#                feedback_entry = {
#                    "applicant_id": selected_idx,
#                    "model_prob": prob,
#                    "human_decision": human_vote,
#                    "reason_codes": reason_codes,
#                    "comments": comments
#                }
#                
#                if 'feedback_log' not in st.session_state:
#                    st.session_state['feedback_log'] = []
#                st.session_state['feedback_log'].append(feedback_entry)
#
#                st.toast("Decision successfully logged to Audit Trail!", icon="💾")
#                st.caption(f"Reasoning captured: *{comments}*")
                
                # Optional: Show 'Agreement' status
                model_bool = 1 if prob > 0.5 else 0
                human_bool = 1 if human_vote == "Approve" else 0
                
                if model_bool == human_bool:
                    st.info("🤝 **Consensus:** Human and AI agree.")
                else:
                    st.warning("⚠️ **Disagreement:** This case is valuable for Model Retraining.")
