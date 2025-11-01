import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import os

# MUST be the first Streamlit command!
st.set_page_config(page_title="Online Food Order Prediction", layout="wide")

# Cache data loading for better performance
@st.cache_data
def load_data():
    """Load and return the dataset"""
    # Try multiple paths for Streamlit Cloud compatibility
    csv_paths = [
        "onlinefoods.csv",  # Same directory as app.py
        "online food delivery app/onlinefoods.csv",  # Subdirectory
        os.path.join(os.path.dirname(__file__), "onlinefoods.csv")  # Relative to script
    ]
    
    for path in csv_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # Return None if file not found (don't call st.error here as it's cached)
    return None

# Load data
data = load_data()


# Streamlit App
st.title("Online Food Order Prediction App")

# Data validation check - must be done before preprocessing
if data is None:
    st.error("Error: Could not find 'onlinefoods.csv'. Please ensure the file exists in the correct location.")
    st.stop()

if data.empty:
    st.error("Error: The dataset is empty. Please check 'onlinefoods.csv'.")
    st.stop()

# Check for required columns before preprocessing
required_columns = ["Age", "Gender", "Marital Status", "Occupation", "Monthly Income", 
                   "Educational Qualifications", "Family size", "Pin code", "Feedback", "Output"]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Error: Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# Preprocessing
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Marital Status"] = data["Marital Status"].map({"Married": 2, "Single": 1, "Prefer not to say": 0})
data["Occupation"] = data["Occupation"].map({"Student": 1, "Employee": 2, "Self Employeed": 3, "House wife": 4})
data["Educational Qualifications"] = data["Educational Qualifications"].map({
    "Graduate": 1, "Post Graduate": 2, "Ph.D": 3, "School": 4, "Uneducated": 5
})

# Add Feedback mapping
data["Feedback"] = data["Feedback"].map({"Positive": 1, "Negative": 0})

# Handle income ranges by converting them to numeric values
income_mapping = {
    "No Income": 0,
    "Below Rs.10000": 5000,
    "Rs.10000 - Rs.25000": 17500,
    "Rs.25000 - Rs.50000": 37500,
    "Above Rs.50000": 75000,
    "More than 50000": 75000,
    "10001 to 25000": 17500,
    "25001 to 50000": 37500
}
data["Monthly Income"] = data["Monthly Income"].replace(income_mapping).infer_objects(copy=False)
data["Monthly Income"] = pd.to_numeric(data["Monthly Income"])

# Handle missing values
# Fill numeric columns with median
numeric_columns = ["Age", "Monthly Income", "Family size", "Pin code"]
for col in numeric_columns:
    data[col] = data[col].fillna(data[col].median())

# Fill categorical columns with mode
categorical_columns = ["Gender", "Marital Status", "Occupation", "Educational Qualifications", "Feedback"]
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Remove any remaining rows with NaN values
data = data.dropna()

# Cache model training for better performance
@st.cache_resource
def train_model(data):
    """Train and return the prediction model"""
    # Prepare features and target
    x = np.array(data[["Age", "Gender", "Marital Status", "Occupation", 
                       "Monthly Income", "Educational Qualifications", 
                       "Family size", "Pin code", "Feedback"]])
    y = np.array(data["Output"]).ravel()  # Convert to 1D array
    
    # Train model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(x, y)
    return model

try:
    model = train_model(data)
except Exception as e:
    st.error(f"Error during model training: {str(e)}")
    st.stop()

# Sidebar for user input
try:
    st.sidebar.header("Customer Details")
    with st.sidebar.form("prediction_form"):
        a = st.number_input("Enter the Age of the Customer", min_value=0, max_value=100, value=25)
        b = st.selectbox("Enter the Gender of the Customer", 
                        options=[("Male", 1), ("Female", 0)], 
                        format_func=lambda x: x[0])[1]
        c = st.selectbox("Marital Status of the Customer", 
                        options=[("Single", 1), ("Married", 2), ("Not Revealed", 0)], 
                        format_func=lambda x: x[0])[1]
        d = st.selectbox("Occupation of the Customer", 
                        options=[("Student", 1), ("Employee", 2), ("Self Employeed", 3), ("House wife", 4)], 
                        format_func=lambda x: x[0])[1]
        e = st.number_input("Monthly Income", min_value=0, value=50000)
        f = st.selectbox("Educational Qualification", 
                        options=[("Graduate", 1), ("Post Graduate", 2), ("Ph.D", 3), ("School", 4), ("Uneducated", 5)], 
                        format_func=lambda x: x[0])[1]
        g = st.number_input("Family Size", min_value=1, max_value=20, value=4)
        h = st.number_input("Pin Code", min_value=0, value=560001)
        i = st.selectbox("Review of the Last Order", 
                        options=[("Positive", 1), ("Negative", 0)], 
                        format_func=lambda x: x[0])[1]
        
        predict_button = st.form_submit_button("Predict")

    # Predict button
    if predict_button:
        try:
            features = np.array([[a, b, c, d, e, f, g, h, i]])
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)[0]
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Main prediction result with custom styling
                st.markdown("""
                    <style>
                    .prediction-box {
                        padding: 20px;
                        border-radius: 10px;
                        margin: 10px 0;
                        text-align: center;
                    }
                    .positive {
                        background-color: #90EE90;
                        color: #006400;
                    }
                    .negative {
                        background-color: #FFB6C1;
                        color: #8B0000;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                if prediction[0] == 'Yes':
                    st.markdown(f"""
                        <div class="prediction-box positive">
                            <h2>Customer Will Order Again! 🎉</h2>
                            <h3>Confidence: {prediction_proba[1]*100:.1f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-box negative">
                            <h2>Customer May Not Order Again ⚠️</h2>
                            <h3>Confidence: {prediction_proba[0]*100:.1f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)

            with col2:
                # Probability gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction_proba[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probability of Ordering Again"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightcoral"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig)

            # Feature importance analysis
            st.markdown("### 📊 Feature Analysis")
            feature_names = ["Age", "Gender", "Marital Status", "Occupation", 
                           "Monthly Income", "Education", "Family Size", 
                           "Pin Code", "Last Order Feedback"]
            
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)

            fig = px.bar(feature_importance, 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title='Feature Importance Analysis',
                        color='importance',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig)

            # Customer Profile Summary
            st.markdown("### 👤 Customer Profile Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Age Group", 
                         "Young Adult" if a < 30 else "Adult" if a < 50 else "Senior",
                         f"{a} years")
            with col2:
                st.metric("Income Level", 
                         "High" if e > 50000 else "Medium" if e > 25000 else "Low",
                         f"₹{e:,}")
            with col3:
                st.metric("Family Size", 
                         "Large" if g > 4 else "Medium" if g > 2 else "Small",
                         f"{g} members")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

except Exception as e:
    st.error(f"Error in form input: {str(e)}")

