import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

st.set_page_config(page_title="Data Quality Audit", layout="wide")

# 1. SESSION STATE & DATA SYNC
if 'X' not in st.session_state:
    st.warning("Please go to Home first!")
    st.stop()

# Initialize alignment tracking for the Global Tab
if 'alignment_history' not in st.session_state:
    st.session_state.alignment_history = {"Agree": 820, "Disagree": 74}

X_full = st.session_state['X']
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0

# Sidebar Selection
st.session_state['selected_index'] = st.sidebar.selectbox(
    "Select Applicant Index", 
    options=range(len(st.session_state['X_test'])),
    index=st.session_state['selected_index']
)

selected_idx = st.session_state['selected_index']
applicant_data = st.session_state['X_test'].iloc[selected_idx]

# 2. HEADER
st.title("🛡️ Data Audit Control")
st.caption(f"Auditing Applicant ID: {selected_idx} | Phase: Human-in-the-Loop Verification")

tab_global, tab_local, tab_alerts, = st.tabs(['Model Fine-Tuning', 'Raw Data', 'Data Alerts'])

# 3. TAB 1: GLOBAL MODEL HEALTH & FEEDBACK LOOP
with tab_global:
    st.write("")
    st.write("Your feedback is the most valuable signal for improving the model. Disagreements are used to re-weight feature importance in the next training cycle.")
    st.info("""
        **Why your feedback matters:**
        1.  **Edge Case Detection:** You spot context (e.g., "Policy Exception") that the model misses.
        2.  **Bias Correction:** Human review prevents statistical anomalies from becoming bias.
        3.  **Active Learning:** Disagreements are flagged as 'High Value' training examples.
        """)
     
    col_text, gap, col_viz = st.columns([2, 0.5, 1.5])
    with col_text:

        
        st.markdown("**Pending Fine-Tuning Queue:**")
        st.table(pd.DataFrame({
            "Trigger Type": ["Human Disagreement", "Drift Detected", "Policy Update"],
            "Cases Count": [12, 5, 1],
            "Status": ["Ready for Retraining", "Analyzing", "Pending"]
        }))


    with col_viz:
        # Precision Growth Chart
        feedback_pts = np.array([0, 100, 200, 300, 400, 500])
        acc_pts = np.array([20, 80, 70, 50, 83, 91])
        fig_learn, ax_learn = plt.subplots(figsize=(6, 3.5))
        ax_learn.plot(feedback_pts, acc_pts, marker='o', color='#1f77b4', linewidth=2)
        ax_learn.fill_between(feedback_pts, acc_pts, alpha=0.1, color='#1f77b4')
        ax_learn.set_title("Impact of Expert Feedback on Accuracy", fontsize=10)
        ax_learn.set_xlabel("Number of Expert Audits Submitted")
        ax_learn.set_ylabel("Model Accuracy (%)")
        st.pyplot(fig_learn)
        plt.close(fig_learn)



# 4. TAB 2: RAW DATA CHECK
with tab_local:
    st.write("")
    st.caption("Compare the applicant's specific raw values against the wider population to spot data entry errors or outliers.")
    st.write("")
    st.write("Applicant transactional data is automatically fetched from Rabo Bank only and it contains 19 568 transaction from Jan-2025 to Jun-2025 (6 complete months).")
    st.write("")
    
    numeric_cols = X_full.select_dtypes(include=np.number).columns.tolist()
    target_feature = st.selectbox("Select Feature to Inspect:", options=numeric_cols, index=0)

    # Stats
    app_val = applicant_data[target_feature]
    avg = X_full[target_feature].mean()
    percentile = (X_full[target_feature] < app_val).mean() * 100
    z_score = (app_val - avg) / X_full[target_feature].std()

    col_viz, col_metrics = st.columns([2, 1])

    with col_viz:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        # Histogram of population
        ax_hist.hist(X_full[target_feature].dropna(), bins=40, color='#e6e9ef', edgecolor='white', label='Population')
        # Line for applicant
        ax_hist.axvline(app_val, color='#ff4b4b', linestyle='-', linewidth=3, label=f'Applicant ({app_val:.2f})')
        # Line for Average
        ax_hist.axvline(avg, color='blue', linestyle='--', linewidth=1, label=f'Average ({avg:.2f})')
        
        ax_hist.set_title(f"Distribution of {target_feature}")
        ax_hist.legend()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with col_metrics:
        st.markdown(f"**Applicant Value:**")
        st.markdown(f"## {app_val:,.2f}")
        
        st.markdown(f"**vs Population:**")
        st.metric("Percentile", f"{percentile:.1f}%", help="Higher means this applicant is above most others.")
        
        if abs(z_score) > 3:
            st.error(f"⚠️ **Extreme Outlier**\n\n(Z-Score: {z_score:.1f}) Verify source document.")
        elif abs(z_score) > 2:
            st.warning(f"⚠️ **High Deviation**\n\n(Z-Score: {z_score:.1f}) Check for data entry error.")
        else:
            st.success(f"✅ **Normal Range**\n\n(Z-Score: {z_score:.1f}) Data looks valid.")



# 5. TAB 3: (DECISION CHECKS with GEMINI)
with tab_alerts:
    
    # DEFINING THE DATA
    findings = {
        "tax_status": "Tax Payment Missing",
        "salary_status": "Salary Payment Delays",
        "identity_status": "Frequent Cash Withdrawal"
    }

    # HELPER FUNCTIONS (CALLBACKS)
    # These execute BEFORE the page reloads to ensure state is updated safely.

    def load_audit_for_editing(idx, log_data):
        """Callback: Loads historical data back into the active form."""
        for f_key, f_data in log_data['details'].items():
            st.session_state[f"decision_{f_key}"] = f_data['decision']
            st.session_state[f"note_val_{f_key}"] = f_data['note']
        st.session_state.editing_idx = idx

    def delete_audit_entry(idx):
        """Callback: Deletes an entry safely."""
        # 1. Safety Check: Ensure index is valid
        if 0 <= idx < len(st.session_state.audit_log):
            st.session_state.audit_log.pop(idx)
            
            # 2. Handle Edit Mode Conflicts
            if st.session_state.editing_idx == idx:
                st.session_state.editing_idx = None # Cancel editing if deleted the active item
            elif st.session_state.editing_idx is not None and st.session_state.editing_idx > idx:
                st.session_state.editing_idx -= 1 # Shift index down if deleted an item above it
            
            st.toast("Entry deleted.", icon="🗑️")

    def submit_audit_entry(findings_dict):
        """Callback: Validates, Saves, and Clears inputs."""
        # 1. Capture Data from Widgets
        snapshot = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details": {}
        }
        
        all_filled = True
        for k in findings_dict.keys():
            dec = st.session_state.get(f"decision_{k}")
            note = st.session_state.get(f"note_val_{k}")
            snapshot["details"][k] = {"decision": dec, "note": note}
            if not dec: 
                all_filled = False

        # 2. Logic: Save or Update
        if all_filled:
            if st.session_state.editing_idx is not None:
                # Update Existing Entry
                idx = st.session_state.editing_idx
                st.session_state.audit_log[idx] = snapshot
                st.session_state.editing_idx = None # Exit edit mode
                st.toast("Audit entry updated successfully.", icon="✅")
            else:
                # Create New Entry
                st.session_state.audit_log.append(snapshot)
                st.session_state.alignment_history["Agree"] += 1 
                st.toast("New audit logged to system.", icon="💾")

            # 3. Clear Inputs (Safe to do here)
            for k in findings_dict.keys():
                st.session_state[f"decision_{k}"] = None
                st.session_state[f"note_val_{k}"] = ""
        else:
            # Validation Error
            st.toast("Please complete all decision fields.", icon="⚠️")

    # 1. SETUP SESSION STATE
    if 'audit_log' not in st.session_state: st.session_state.audit_log = []
    if 'editing_idx' not in st.session_state: st.session_state.editing_idx = None 

    st.write("")
    st.write("These flags were raised by the automated pre-screening system. Please resolve them before final approval.")
    st.write("")

    # 2. INPUT FORM
    st.markdown("### New Audit Entry" if st.session_state.editing_idx is None else f"### Editing Entry #{st.session_state.editing_idx + 1}")
    
    with st.container():
        for key, label in findings.items():
            st.markdown(f"**{label}**")
            c_dec, c_note = st.columns([1, 2])
            
            # Initialize keys if missing (First Run)
            if f"decision_{key}" not in st.session_state: st.session_state[f"decision_{key}"] = None
            if f"note_val_{key}" not in st.session_state: st.session_state[f"note_val_{key}"] = ""
            
            with c_dec:
                st.segmented_control(
                    "Decision:", 
                    ["✅ Agree", "❌ Disagree", "⏳ Review"], 
                    key=f"decision_{key}", 
                    label_visibility="collapsed"
                )
            with c_note:
                st.text_area(
                    "Note:", 
                    placeholder="Justification...", 
                    key=f"note_val_{key}", 
                    label_visibility="collapsed", 
                    height=68
                )

    st.write("")
    
    # 3. SUBMIT BUTTON
    _, col_submit = st.columns([4, 1.5])
    with col_submit:
        btn_label = "🔄 Update Entry" if st.session_state.editing_idx is not None else "💾 Resolve & Log"
        
        # On_click to trigger the save function BEFORE the page reloads
        st.button(
            btn_label, 
            use_container_width=True, 
            type="primary",
            on_click=submit_audit_entry,
            args=(findings,)
        )

    st.divider()

    # 4. HISTORY SECTION
    st.markdown("### Audit History")

    if not st.session_state.audit_log:
        st.caption("No audits logged for this session yet.")
    else:
        # Loop through logs in reverse order (newest first)
        for i, log in reversed(list(enumerate(st.session_state.audit_log))):
            
            with st.container():
                # Header: Timestamp + Edit/Delete Controls
                c_time, c_acts = st.columns([4, 1])
                with c_time:
                    st.markdown(f"**Audit #{i+1}** <span style='color:grey; font-size:0.8em'> | {log['timestamp']}</span>", unsafe_allow_html=True)
                
                with c_acts:
                    col_e, col_d = st.columns(2)
                    with col_e:
                        st.button("✏️", key=f"edit_{i}", on_click=load_audit_for_editing, args=(i, log), help="Edit")
                    with col_d:
                        st.button("🗑️", key=f"del_{i}", on_click=delete_audit_entry, args=(i,), help="Delete")

                # Body: Decisions & Notes
                for f_key, f_data in log['details'].items():
                    dec_str = f_data['decision'] if f_data['decision'] else "❓"
                    icon = dec_str.split(" ")[0] # Grab just the emoji
                    note_str = f" - *{f_data['note']}*" if f_data['note'] else ""
                    
                    st.write(f"- **{findings[f_key]}**: {icon} {note_str}")
                
                st.markdown("---")





### 5. TAB 3: DECISION CHECKS
##with tab_alerts:
##    st.write("")
##    st.write("These flags were raised by the automated pre-screening system. Please resolve them before final approval.")
##    st.write("")
##
##    findings = {
##        "tax_status": "Tax Payment Missing",
##        "salary_status": "Salary Payment Delays",
##        "identity_status": "Frequent Cash Withdrawal"
##    }
##    st.write("")
##    
##    for key, label in findings.items():
##        st.markdown(f"**{label}**")
##        c_dec, c_note = st.columns([1, 2])
##        with c_dec:
##            st.segmented_control("Decision:", ["✅ Agree", "❌ Disagree", "⏳ Review"], key=f"decision_{key}", label_visibility="collapsed")
##        with c_note:
##            st.text_area("Note:", placeholder="Justification...", key=f"note_val_{key}", label_visibility="collapsed", height=68)
##
##    st.write("")
##    st.divider()
##   
##    # Submission Logic connects back to Global Tab
##    _, col_submit = st.columns([4, 1.5])
##    with col_submit:
##        if st.button("💾 Resolve & Update Model", use_container_width=True, type="primary"):
##            # Update the global alignment metric simulation
##            # (In a real app, you'd check if Human agreed with the 'Alert' or dismissed it)
##            st.session_state.alignment_history["Agree"] += 1 
##            st.balloons()
##            st.success("Alerts Sent")
