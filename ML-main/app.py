import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Marketing AI Predictor", layout="wide")

@st.cache_resource
def load_models():
    reg_model = joblib.load('regression_model.pkl')
    clf_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return reg_model, clf_model, scaler

try:
    with st.spinner("Loading models... Please wait..."):
        reg_model, clf_model, scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}\n\nPlease ensure model files (.pkl) are uploaded in the same directory and scikit-learn is updated.")
    st.stop() 

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Page:", 
                            ["Data and Training Overview", 
                             "1. Campaign Conversions (Regression)", 
                             "2. User Conversion (Classification)"])

# =========================================================================
# Data and Training Overview Dashboard
# =========================================================================
if app_mode == "Data and Training Overview":
    st.title("Project Data and Training Overview")
    st.markdown("An interactive overview of the datasets, exploratory data analysis (EDA), and the final model training results for both tasks.")
    
    @st.cache_data
    def load_all_data():
        # Load Classification Data
        df_clf = pd.read_csv('digital_marketing_campaign_dataset.csv')
        
        # Load and Merge Regression Data
        nykaa = pd.read_csv('nykaa_campaign_data.csv')
        purplle = pd.read_csv('purplle_campaign_data.csv')
        tira = pd.read_csv('tira_campaign_data.csv')
        
        nykaa["Brand"] = "Nykaa"
        purplle["Brand"] = "Purplle"
        tira["Brand"] = "Tira"
        
        df_reg = pd.concat([nykaa, purplle, tira], ignore_index=True)
        
        return df_reg, df_clf
        
    try:
        df_reg, df_clf = load_all_data()
        
        tab1, tab2 = st.tabs(["Regression Task (Campaign Conversions)", "Classification Task (User Conversion)"])
        
        # ---------------------------------------------------------
        # TAB 1: REGRESSION
        # ---------------------------------------------------------
        with tab1:
            st.header("Campaign Conversions Analysis")
            st.markdown("Data merged from Nykaa, Purplle, and Tira datasets.")
            
            st.subheader("Raw Data Sample")
            st.dataframe(df_reg.head(10))
            
            # Metrics
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Total Records", f"{len(df_reg):,}")
            r_col2.metric("Average Conversions", f"{df_reg['Conversions'].mean():.0f}")
            r_col3.metric("Total Revenue", f"${df_reg['Revenue'].sum():,.0f}")
            
            st.markdown("---")
            st.subheader("Exploratory Data Analysis")
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**Correlation Matrix**")
                fig_corr_reg, ax_corr_reg = plt.subplots(figsize=(8, 6))
                sns.heatmap(df_reg.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax_corr_reg)
                st.pyplot(fig_corr_reg)
                
            with p_col2:
                st.markdown("**Conversions Distribution**")
                fig_dist_reg, ax_dist_reg = plt.subplots(figsize=(8, 6))
                sns.histplot(df_reg["Conversions"], bins=50, kde=True, color="salmon", ax=ax_dist_reg)
                st.pyplot(fig_dist_reg)
            
            st.markdown("---")
            st.subheader("Model Evaluation & Feature Importance")
            
            m_col1, m_col2 = st.columns([1, 1])
            with m_col1:
                st.markdown("**Cross-Validation & Test Results**")
                reg_results = pd.DataFrame({
                    "Model": ["Random Forest", "Decision Tree", "Ridge Regression", "Baseline (Mean)"],
                    "R2 Score": ["0.8381", "0.8371", "0.0831", "-0.1224"],
                    "CV R2 (mean)": ["0.9032", "0.9029", "0.7537", "0.0000"],
                    "RMSE": ["345.36", "346.40", "821.95", "909.39"],
                    "MAE": ["236.55", "236.90", "376.01", "618.63"]
                }).set_index("Model")
                st.dataframe(reg_results, use_container_width=True)
                
            with m_col2:
                st.markdown("**Random Forest Feature Importance**")
                try:
                    feature_names = reg_model.named_steps["prep"].get_feature_names_out()
                    importances = reg_model.named_steps["model"].feature_importances_
                    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                    imp_df = imp_df.sort_values("Importance", ascending=False).head(10)
                    
                    fig_imp_reg, ax_imp_reg = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis", ax=ax_imp_reg)
                    st.pyplot(fig_imp_reg)
                except Exception:
                    st.warning("Feature importance not available for this model configuration.")

        # ---------------------------------------------------------
        # TAB 2: CLASSIFICATION
        # ---------------------------------------------------------
        with tab2:
            st.header("User Conversion Analysis")
            st.markdown("Data loaded from the digital marketing campaign dataset.")
            
            st.subheader("Raw Data Sample")
            st.dataframe(df_clf.head(10))
            
            # Metrics
            c_col1, c_col2, c_col3 = st.columns(3)
            c_col1.metric("Total Records", f"{len(df_clf):,}")
            c_col2.metric("Average Conversion Rate", f"{(df_clf['ConversionRate'].mean() * 100):.2f}%")
            c_col3.metric("Total Converted Users", f"{df_clf['Conversion'].sum():,}")
            
            st.markdown("---")
            st.subheader("Exploratory Data Analysis")
            
            p_col3, p_col4 = st.columns(2)
            with p_col3:
                st.markdown("**Target Variable Distribution (Imbalance)**")
                fig_dist_clf, ax_dist_clf = plt.subplots(figsize=(8, 6))
                sns.countplot(data=df_clf, x='Conversion', palette="Blues_d", ax=ax_dist_clf)
                ax_dist_clf.set_xticklabels(['Not Converted (0)', 'Converted (1)'])
                st.pyplot(fig_dist_clf)
                
            with p_col4:
                st.markdown("**Numeric Features Correlation Matrix**")
                fig_corr_clf, ax_corr_clf = plt.subplots(figsize=(8, 6))
                numeric_cols = df_clf.select_dtypes(include=[np.number])
                sns.heatmap(numeric_cols.corr(), annot=False, cmap="coolwarm", ax=ax_corr_clf)
                st.pyplot(fig_corr_clf)
            
            st.markdown("---")
            st.subheader("Model Evaluation & Feature Importance")
            
            m_col3, m_col4 = st.columns([1, 1])
            with m_col3:
                st.markdown("**Test Results (Balanced Class Weights applied)**")
                clf_results = pd.DataFrame({
                    "Model": ["Random Forest", "SVM", "Decision Tree", "Logistic Regression"],
                    "F1 Score": ["0.9545", "0.9046", "0.8533", "0.8429"],
                    "ROC-AUC": ["0.8150", "0.7874", "0.6624", "0.7816"],
                    "Accuracy": ["0.9181", "0.8388", "0.7600", "0.7531"],
                    "Precision": ["0.9308", "0.9393", "0.9186", "0.9524"]
                }).set_index("Model")
                st.dataframe(clf_results, use_container_width=True)
                
            with m_col4:
                st.markdown("**Random Forest Feature Importance**")
                try:
                    if hasattr(clf_model, "feature_importances_"):
                        X_cols = pd.get_dummies(df_clf.drop(columns=["Conversion", "CustomerID", "AdvertisingPlatform", "AdvertisingTool"]), drop_first=True).columns
                        importances_clf = clf_model.feature_importances_
                        imp_df_clf = pd.DataFrame({"Feature": X_cols, "Importance": importances_clf})
                        imp_df_clf = imp_df_clf.sort_values("Importance", ascending=False).head(10)
                        
                        fig_imp_clf, ax_imp_clf = plt.subplots(figsize=(8, 5))
                        sns.barplot(data=imp_df_clf, x="Importance", y="Feature", palette="magma", ax=ax_imp_clf)
                        st.pyplot(fig_imp_clf)
                    else:
                        st.info("Feature importance plot is available for Tree-based models.")
                except Exception:
                    st.warning("Feature importance not available for this model configuration.")
                    
    except FileNotFoundError:
        st.warning("One or more CSV files are missing. Please ensure 'digital_marketing_campaign_dataset.csv', 'nykaa_campaign_data.csv', 'purplle_campaign_data.csv', and 'tira_campaign_data.csv' are uploaded to the project directory.")

# =========================================================================
# Model 1: Regression (Campaign Conversions)
# =========================================================================
elif app_mode == "1. Campaign Conversions (Regression)":
    st.title("Campaign Conversions Predictor")
    st.markdown("Enter the campaign details below to predict the number of conversions.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
        campaign_type = st.selectbox("Campaign Type", ["Influencer", "Email", "Paid Ads", "SEO", "Social Media"])
        target_audience = st.selectbox("Target Audience", ["Premium Shoppers", "Working Women", "Youth", "Tier 2 City Customers", "College Students"])
        customer_segment = st.selectbox("Customer Segment", ["Youth", "College Students", "Working Women", "Tier 2 City Customers", "Premium Shoppers"])
        language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Bengali"])
        
    with col2:
        impressions = st.number_input("Impressions", min_value=0, value=50000)
        clicks = st.number_input("Clicks", min_value=0, value=5000)
        leads = st.number_input("Leads", min_value=0, value=1000)
        duration = st.number_input("Duration (Days)", min_value=1, value=15)
        year = st.number_input("Year", min_value=2000, value=2024, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        
    with col3:
        st.write("Channels Used (Check all that apply):")
        facebook = st.checkbox("Facebook")
        whatsapp = st.checkbox("WhatsApp")
        google = st.checkbox("Google")
        youtube = st.checkbox("YouTube")
        instagram = st.checkbox("Instagram")
        email = st.checkbox("Email")
        
    if st.button("Predict Conversions"):
        input_data = pd.DataFrame({
            "Impressions": [impressions],
            "Clicks": [clicks],
            "Leads": [leads],
            "Brand": [brand],
            "Campaign_Type": [campaign_type],
            "Target_Audience": [target_audience],
            "Language": [language],
            "Customer_Segment": [customer_segment],
            "Duration": [duration],
            "Year": [year],
            "Month": [month],
            "Facebook": [int(facebook)],
            "WhatsApp": [int(whatsapp)],
            "Google": [int(google)],
            "YouTube": [int(youtube)],
            "Instagram": [int(instagram)],
            "Email": [int(email)],
        })
        
        log_pred = reg_model.predict(input_data)[0]
        actual_pred = np.expm1(log_pred)
        
        # حساب نسبة التحويل وتجنب القسمة على صفر
        conversion_rate = (actual_pred / leads) * 100 if leads > 0 else 0
        
        st.success(f"Predicted Conversions: {int(actual_pred):,}")
        st.info(f"Expected Conversion Rate (from Leads): {conversion_rate:.2f}%")

# =========================================================================
# Model 2: Classification (User Conversion)
# =========================================================================
elif app_mode == "2. User Conversion (Classification)":
    st.title("User Conversion Classifier")
    st.markdown("Enter user and campaign metrics to predict if they will convert (1) or not (0).")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        income = st.number_input("Income ($)", min_value=0, value=50000)
        ad_spend = st.number_input("Ad Spend ($)", min_value=0.0, value=1000.0)
        ctr = st.number_input("Click-Through Rate", min_value=0.0, max_value=1.0, value=0.1)
        cvr = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.05)
        
    with col2:
        visits = st.number_input("Website Visits", min_value=0, value=5)
        pages = st.number_input("Pages Per Visit", min_value=0.0, value=3.0)
        time_site = st.number_input("Time On Site (mins)", min_value=0.0, value=5.0)
        shares = st.number_input("Social Shares", min_value=0, value=10)
        email_opens = st.number_input("Email Opens", min_value=0, value=2)
        email_clicks = st.number_input("Email Clicks", min_value=0, value=1)
        
    with col3:
        prev_purchases = st.number_input("Previous Purchases", min_value=0, value=1)
        loyalty = st.number_input("Loyalty Points", min_value=0, value=500)
        campaign_channel = st.selectbox("Campaign Channel", ["Email", "PPC", "Referral", "SEO", "Social Media"])
        campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
        
    if st.button("Predict Conversion Status"):
        
        clf_input = {
            'Age': age,
            'Gender': 1 if gender == 'Male' else 0,
            'Income': income,
            'AdSpend': ad_spend,
            'ClickThroughRate': ctr,
            'ConversionRate': cvr,
            'WebsiteVisits': visits,
            'PagesPerVisit': pages,
            'TimeOnSite': time_site,
            'SocialShares': shares,
            'EmailOpens': email_opens,
            'EmailClicks': email_clicks,
            'PreviousPurchases': prev_purchases,
            'LoyaltyPoints': loyalty,
            
            'CampaignChannel_PPC': 1 if campaign_channel == 'PPC' else 0,
            'CampaignChannel_Referral': 1 if campaign_channel == 'Referral' else 0,
            'CampaignChannel_SEO': 1 if campaign_channel == 'SEO' else 0,
            'CampaignChannel_Social Media': 1 if campaign_channel == 'Social Media' else 0,
            
            'CampaignType_Consideration': 1 if campaign_type == 'Consideration' else 0,
            'CampaignType_Conversion': 1 if campaign_type == 'Conversion' else 0,
            'CampaignType_Retention': 1 if campaign_type == 'Retention' else 0,
        }
        
        df_clf = pd.DataFrame([clf_input])
        
        scaled_data = scaler.transform(df_clf)
        
        prediction = clf_model.predict(scaled_data)[0]
        
        if hasattr(clf_model, "predict_proba"):
            prob = clf_model.predict_proba(scaled_data)[0][1]
            prob_text = f"(Probability: {prob:.2%})"
        else:
            prob_text = ""
            
        if prediction == 1:
            st.success(f"Result: Converted (Yes) {prob_text}")
        else:
            st.error(f"Result: Not Converted (No) {prob_text}")
