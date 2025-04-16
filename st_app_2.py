
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# Cache the data loading and figure creation for performance
@st.cache_data
def load_data_and_create_figure():
    # Load the Titanic dataset
    df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
    age = df['Age'].dropna()  # Drop NaN values for age KDE calculation

    # Create a 2x2 subplot grid with Plotly
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Survival Distribution', 
                                        'Age Distribution', 
                                        'Fare Distribution', 
                                        'Correlation Matrix'),
                        vertical_spacing=0.15,  # Add spacing to avoid overlap
                        horizontal_spacing=0.15)

    # Survival count plot
    survived_counts = df['Survived'].value_counts()
    fig.add_trace(
        go.Bar(
            x=['Did not survive'], 
            y=[survived_counts.get(0, 0)], 
            name='Did not survive', 
            marker=dict(color='red'), 
            text=[survived_counts.get(0, 0)], 
            textposition='auto'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=['Survived'], 
            y=[survived_counts.get(1, 0)], 
            name='Survived', 
            marker=dict(color='blue'), 
            text=[survived_counts.get(1, 0)], 
            textposition='auto'
        ),
        row=1, col=1
    )

    # Age distribution with histogram and KDE
    kde = gaussian_kde(age)
    x = np.linspace(age.min(), age.max(), 100)
    kde_values = kde(x)
    fig.add_trace(
        go.Histogram(x=age, 
                     nbinsx=30, 
                     histnorm='probability density', 
                     name='Age', 
                     marker=dict(color='orange'), 
                     opacity=0.7
                    ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x, y=kde_values, mode='lines', name='KDE', line=dict(color='red')),
        row=1, col=2
    )

    # Fare boxplot
    fig.add_trace(
        go.Box(x=df['Fare'], 
               name='Fare', 
               boxpoints='outliers',
               line=dict(color='lightblue'), 
               fillcolor='lightblue', 
               opacity=0.7
              ),
        row=2, col=1
    )

    # Correlation heatmap with adjusted colorbar
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values, 
            x=corr_matrix.columns, 
            y=corr_matrix.columns, 
            colorscale='RdBu',
            text=corr_matrix.values.round(2), 
            texttemplate="%{text}", 
            textfont={"size": 10},
            colorbar=dict(
                len=0.45, 
                y=0.21, 
                yanchor='middle'
            )
        ),
        row=2, col=2
    )

    # Update layout and axes
    fig.update_layout(
        height=1200, 
        width=1200, 
        title_text="Univariate and Multivariate Analysis",
        title_x=0.5, 
        showlegend=True,
        barmode='group'
    )

    fig.update_xaxes(title_text="Survival Status", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Age", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)
    fig.update_xaxes(title_text="Fare", row=2, col=1)

    return df, fig

# Load data and create the figure
df, fig = load_data_and_create_figure()

# App interface
st.title("Titanic Survival Predictor")

# Display the interactive Plotly figure
st.subheader("EDA")
st.plotly_chart(fig, use_container_width=True)

# Load pre-trained model and preprocessor
model = joblib.load("titanic_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Predictive interface
st.subheader("Model Prediction")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)

with col2:
    fare = st.number_input("Fare", min_value=0, value=50)
    family_size = st.number_input("Family Size", min_value=0, max_value=10, value=0)

# Prediction logic
if st.button("Predict Survival"):
    input_data = pd.DataFrame([[pclass, sex, age, fare, family_size]],
                              columns=['Pclass',
                                       'Sex',
                                       'Age',
                                       'Fare',
                                       'FamilySize'])
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    st.header("Result")
    st.metric("Survival Probability", f"{probability:.1%}")
    st.write(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
