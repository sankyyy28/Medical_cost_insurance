import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess data
@st.cache_data
def load_data():
    # Your dataset here (I'll use a sample, replace with your full data)
    data = """age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,southwest,16884.924
18,male,33.77,1,no,southeast,1725.5523
28,male,33,3,no,southeast,4449.462
33,male,22.705,0,no,northwest,21984.47061
32,male,28.88,0,no,northwest,3866.8552
31,female,25.74,0,no,southeast,3756.6216
46,female,33.44,1,no,southeast,8240.5896
37,female,27.74,3,no,northwest,7281.5056
37,male,29.83,2,no,northeast,6406.4107
60,female,25.84,0,no,northwest,28923.13692"""
    
    df = pd.read_csv(pd.compat.StringIO(data))
    # Add your full dataset here instead of the sample above
    
    # Preprocessing
    df['smoker_yes'] = (df['smoker'] == 'yes').astype(int)
    return df

@st.cache_resource
def train_model(df):
    features = ['bmi', 'age', 'smoker_yes']
    X = df[features]
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, features

# Main app
def main():
    st.title("🏥 Insurance Medical Charges Predictor")
    st.markdown("Predict medical insurance charges based on BMI, Age, and Smoking status")
    
    # Load data and train model
    df = load_data()
    model, scaler, features = train_model(df)
    
    # Sidebar for user input
    st.sidebar.header("🔧 Input Parameters")
    
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=35, step=1)
    bmi = st.sidebar.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    smoker = st.sidebar.selectbox("Smoker", options=["No", "Yes"])
    
    smoker_yes = 1 if smoker == "Yes" else 0
    
    # Prediction
    if st.sidebar.button("Predict Charges"):
        # Prepare input data
        input_data = np.array([[bmi, age, smoker_yes]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        st.sidebar.success(f"**Predicted Insurance Charges:** ${prediction:,.2f}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Dataset Overview")
        
        # Show basic statistics
        st.write("**Dataset Summary:**")
        st.dataframe(df[['age', 'bmi', 'smoker', 'charges']].describe(), use_container_width=True)
        
        # Show sample data
        st.write("**Sample Data (5 rows):**")
        st.dataframe(df[['age', 'bmi', 'smoker', 'charges']].head(), use_container_width=True)
    
    with col2:
        st.subheader("📈 Data Visualization")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Charges Distribution", "Features vs Charges", "Smoker Analysis"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['charges'], bins=30, kde=True, ax=ax)
            ax.set_title('Distribution of Insurance Charges')
            ax.set_xlabel('Charges ($)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with tab2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Age vs Charges
            sns.scatterplot(data=df, x='age', y='charges', hue='smoker', ax=axes[0])
            axes[0].set_title('Age vs Insurance Charges')
            axes[0].set_xlabel('Age')
            axes[0].set_ylabel('Charges ($)')
            
            # BMI vs Charges
            sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', ax=axes[1])
            axes[1].set_title('BMI vs Insurance Charges')
            axes[1].set_xlabel('BMI')
            axes[1].set_ylabel('Charges ($)')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='smoker', y='charges', ax=ax)
            ax.set_title('Insurance Charges by Smoking Status')
            ax.set_xlabel('Smoker')
            ax.set_ylabel('Charges ($)')
            st.pyplot(fig)
    
    # Model information section
    st.subheader("🤖 Model Information")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.write("**Features Used:**")
        for feature in features:
            st.write(f"• {feature}")
        
        st.write("**Model Type:** Linear Regression")
        st.write("**Preprocessing:**")
        st.write("- Smoker encoded as binary (0=No, 1=Yes)")
        st.write("- Features standardized using StandardScaler")
    
    with col4:
        st.write("**Model Coefficients:**")
        coefficients = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        st.dataframe(coefficients, use_container_width=True)
        st.write(f"**Intercept:** ${model.intercept_:.2f}")
    
    # Data statistics
    st.subheader("📋 Data Statistics")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Total Records", len(df))
        st.metric("Average Age", f"{df['age'].mean():.1f} years")
    
    with col6:
        st.metric("Average BMI", f"{df['bmi'].mean():.1f}")
        st.metric("Smokers", f"{df['smoker_yes'].sum()} ({df['smoker_yes'].mean()*100:.1f}%)")
    
    with col7:
        st.metric("Average Charges", f"${df['charges'].mean():,.2f}")
        st.metric("Max Charges", f"${df['charges'].max():,.2f}")

# How to use section
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    1. **Adjust Input Parameters** in the sidebar:
       - Set the Age using the slider
       - Set the BMI (Body Mass Index) using the slider
       - Select whether the person is a Smoker or not
    
    2. **Click "Predict Charges"** to get the predicted insurance cost
    
    3. **Explore the visualizations** to understand the data patterns
    
    4. **Check model information** to see how predictions are made
    """)

# Footer
st.markdown("---")
st.markdown("**Note:** This is a predictive model based on historical insurance data. Actual charges may vary.")

if __name__ == "__main__":
    main()
