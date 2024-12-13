import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

st.set_page_config(layout='wide')

# Custom styling and other Streamlit commands
st.markdown(
    """
    <style>
    /* Custom CSS styles */
    .main {
        background-color: black;
    }

    .stApp {
        background-color: black;
        color: white;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    .css-1aumxhk {
        background-color: #1f1f1f; 
    }

    .stCodeBlock {
        background-color: #333333;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

warnings.filterwarnings("ignore")

# Streamlit Title
st.title("Gym members exercise tracking")

# Tabs for different sections
tabs = st.tabs(["📊 Describe Data", "🔍 EDA and Analysis", "🤖 Prediction"])

# Tab 1: Data Description

with tabs[0]:
    st.header("📊 Describe Data")

    # Read data directly from the CSV file path
    data = pd.read_csv("./gym_members_exercise_tracking.csv")  # Update this path to the correct file location on your system
    st.subheader("Loading Data")
    st.write(data)
    # Data Description
    st.subheader("Data Summary")
    st.write(data.describe().T)

    # Insights
    st.subheader("Key Insights from the Data")
    st.markdown("""
    - **Session Duration**: Analyze the distribution of session durations to identify trends or anomalies.
    - **Calories Burned**: Gain insights by gender, BMI, fat percentage, and workout duration.
    - **Heart Rate**:
        - Maximum BPM (beats per minute) and resting BPM, categorized by gender.
        - Correlations with fat percentage and age.
    - **Experience**:
        - Explore how experience level affects session duration and fat percentage.
    - **Workout Frequency**:
        - Understand how often members work out and how it relates to fat percentage and calories burned.
    """)

    # Key statistics and ranges
    st.subheader("Key Statistics")
    st.markdown(f"""
    - **Age**: Ranges from {data['Age'].min()} to {data['Age'].max()}, with a mean of ~{data['Age'].mean():.2f} years.
    - **Weight**: Ranges from {data['Weight (kg)'].min()} kg to {data['Weight (kg)'].max()} kg.
    - **Height**: Averages ~{data['Height (m)'].mean():.2f} m, with a range from {data['Height (m)'].min()} m to {data['Height (m)'].max()} m.
    - **Calories Burned**: Mean ~{data['Calories_Burned'].mean():.2f} kcal/session.
    - **Workout Frequency**: Averages ~{data['Workout_Frequency (days/week)'].mean():.2f} days per week (range {data['Workout_Frequency (days/week)'].min()}–{data['Workout_Frequency (days/week)'].max()} days).
    - **BMI**: Range {data['BMI'].min():.2f} to {data['BMI'].max():.2f}, with a mean of ~{data['BMI'].mean():.2f}.
    """)

    # Optional visualization (e.g., histogram for session durations)
    st.subheader("Distribution of Session Duration")
    st.bar_chart(data["Session_Duration (hours)"])
# Tab 2: EDA & Data Analysis
with tabs[1]:

        st.header("🔍 EDA & Data Analysis")

        # Group data for some visualizations
        calories_gender = data.groupby('Gender')['Calories_Burned'].mean()
        avg_fat_exp = data.groupby('Experience_Level')['Fat_Percentage'].mean()
        avg_session_exp = data.groupby('Experience_Level')['Session_Duration (hours)'].mean()
        avg_resting_bpm_gender = data.groupby('Gender')['Resting_BPM'].mean()
        avg_max_bpm_workout = data.groupby('Workout_Type')['Max_BPM'].mean()

        data['BMI_Category'] = pd.cut(
            data['BMI'], bins=[0, 18.5, 24.9, 29.9, 40],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        avg_calories_bmi = data.groupby('BMI_Category')['Calories_Burned'].mean()

        # Streamlit Interface
        st.title("Exploratory Data Analysis")
        st.sidebar.header("Select Charts to Display")

        # Options for the user to select
        options = {
            "Session Duration Distribution": st.sidebar.checkbox("Session Duration Distribution"),
            "Calories Burned by Gender": st.sidebar.checkbox("Calories Burned by Gender"),
            "Fat Percentage by Experience Level": st.sidebar.checkbox("Fat Percentage by Experience Level"),
            "Calories Burned vs Session Duration": st.sidebar.checkbox("Calories Burned vs Session Duration"),
            "Max BPM Distribution": st.sidebar.checkbox("Max BPM Distribution"),
            "Average Resting BPM by Gender": st.sidebar.checkbox("Average Resting BPM by Gender"),
            "Workout Frequency Distribution": st.sidebar.checkbox("Workout Frequency Distribution"),
            "Average Max BPM by Workout Type": st.sidebar.checkbox("Average Max BPM by Workout Type"),
            "Water Intake vs Workout Frequency": st.sidebar.checkbox("Water Intake vs Workout Frequency"),
            "Session Duration vs Experience Level": st.sidebar.checkbox("Session Duration vs Experience Level"),
            "Calories Burned by BMI Category": st.sidebar.checkbox("Calories Burned by BMI Category"),
            "Fat Percentage Distribution": st.sidebar.checkbox("Fat Percentage Distribution"),
            "Calories Burned by Fat Percentage": st.sidebar.checkbox("Calories Burned by Fat Percentage"),
            "Workout Frequency vs Fat Percentage": st.sidebar.checkbox("Workout Frequency vs Fat Percentage"),
            "Age vs Max BPM": st.sidebar.checkbox("Age vs Max BPM"),
            "Water Intake vs Session Duration": st.sidebar.checkbox("Water Intake vs Session Duration"),
        }

        # Create a Plotly Subplot
        fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=tuple(options.keys())
        )

        # Add plots conditionally
        row, col = 1, 1
        for chart, selected in options.items():
            if selected:
                if chart == "Session Duration Distribution":
                    fig.add_trace(
                        go.Histogram(x=data['Session_Duration (hours)'], nbinsx=20, marker_color='yellow'),
                        row=row, col=col
                    )
                elif chart == "Calories Burned by Gender":
                    fig.add_trace(
                        go.Bar(x=calories_gender.index, y=calories_gender.values, marker_color='blue'),
                        row=row, col=col
                    )
                elif chart == "Fat Percentage by Experience Level":
                    fig.add_trace(
                        go.Bar(x=avg_fat_exp.index, y=avg_fat_exp.values, marker_color='purple'),
                        row=row, col=col
                    )
                elif chart == "Calories Burned vs Session Duration":
                    fig.add_trace(
                        go.Scatter(
                            x=data['Session_Duration (hours)'], y=data['Calories_Burned'], mode='markers',
                            marker=dict(color='green', size=8)
                        ),
                        row=row, col=col
                    )
                elif chart == "Max BPM Distribution":
                    fig.add_trace(
                        go.Histogram(x=data['Max_BPM'], nbinsx=20, marker_color='orange'),
                        row=row, col=col
                    )
                elif chart == "Average Resting BPM by Gender":
                    fig.add_trace(
                        go.Bar(x=avg_resting_bpm_gender.index, y=avg_resting_bpm_gender.values, marker_color='red'),
                        row=row, col=col
                    )
                elif chart == "Workout Frequency Distribution":
                    fig.add_trace(
                        go.Histogram(x=data['Workout_Frequency (days/week)'], nbinsx=10, marker_color='skyblue'),
                        row=row, col=col
                    )
                elif chart == "Average Max BPM by Workout Type":
                    fig.add_trace(
                        go.Bar(x=avg_max_bpm_workout.index, y=avg_max_bpm_workout.values, marker_color='blue'),
                        row=row, col=col
                    )
                elif chart == "Water Intake vs Workout Frequency":
                    fig.add_trace(
                        go.Scatter(
                            x=data['Workout_Frequency (days/week)'], y=data['Water_Intake (liters)'], mode='markers',
                            marker=dict(color='teal', size=8)
                        ),
                        row=row, col=col
                    )
                elif chart == "Session Duration vs Experience Level":
                    fig.add_trace(
                        go.Bar(x=avg_session_exp.index, y=avg_session_exp.values, marker_color='purple'),
                        row=row, col=col
                    )
                elif chart == "Calories Burned by BMI Category":
                    fig.add_trace(
                        go.Bar(x=avg_calories_bmi.index, y=avg_calories_bmi.values, marker_color='green'),
                        row=row, col=col
                    )
                elif chart == "Fat Percentage Distribution":
                    fig.add_trace(
                        go.Histogram(x=data['Fat_Percentage'], nbinsx=20, marker_color='cyan'),
                        row=row, col=col
                    )
                elif chart == "Calories Burned by Fat Percentage":
                    fig.add_trace(
                        go.Scatter(
                            x=data['Fat_Percentage'], y=data['Calories_Burned'], mode='markers',
                            marker=dict(color='orange', size=8)
                        ),
                        row=row, col=col
                    )
                elif chart == "Workout Frequency vs Fat Percentage":
                    fig.add_trace(
                        go.Scatter(
                            x=data['Workout_Frequency (days/week)'], y=data['Fat_Percentage'], mode='markers',
                            marker=dict(color='green', size=8)
                        ),
                        row=row, col=col
                    )
                elif chart == "Age vs Max BPM":
                    fig.add_trace(
                        go.Scatter(
                            x=data['Age'], y=data['Max_BPM'], mode='markers',
                            marker=dict(color='purple', size=8)
                        ),
                        row=row, col=col
                    )
                elif chart == "Water Intake vs Session Duration":
                    fig.add_trace(
                        go.Scatter(
                            x=data['Water_Intake (liters)'], y=data['Session_Duration (hours)'], mode='markers',
                            marker=dict(color='purple', size=8)
                        ),
                        row=row, col=col
                    )

                # Move to the next subplot cell
                col += 1
                if col > 4:
                    col = 1
                    row += 1

        # Update figure layout
        fig.update_layout(
            title_text="Exploratory Data Analysis",
            height=1600, width=1400,  # Adjusted for wider figures
            showlegend=False,
            font=dict(color='white'),
            template='plotly_dark',
            plot_bgcolor='black',
            paper_bgcolor='black',
        )

        # Display the figure
        st.plotly_chart(fig)
                # Add Session Index
        data['Session_Index'] = range(1, len(data) + 1)

        # Aggregate data by session index
        session_data = data.groupby('Session_Index').agg(
            total_calories=('Calories_Burned', 'sum'),
            max_bpm=('Max_BPM', 'mean'),
            avg_session_duration=('Session_Duration (hours)', 'mean')
        ).reset_index()

        # Streamlit Interface
        st.title("Time Series Analysis by Session Index")
        st.sidebar.header("Select Plots to Display")

        # Sidebar options
        show_total_calories = st.sidebar.checkbox("Total Calories Burned by Session Index")
        show_max_bpm = st.sidebar.checkbox("Max BPM by Session Index")
        show_avg_session_duration = st.sidebar.checkbox("Average Session Duration by Session Index")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Total Calories Burned by Session Index",
                "Max BPM by Session Index",
                "Average Session Duration by Session Index"
            ),
            shared_xaxes=True
        )

        # Add plots conditionally
        row = 1
        if show_total_calories:
            fig.add_trace(
                go.Scatter(
                    x=session_data['Session_Index'],
                    y=session_data['total_calories'],
                    mode='lines+markers',
                    name="Total Calories Burned",
                    marker=dict(color='blue')
                ),
                row=row, col=1
            )
            row += 1

        if show_max_bpm:
            fig.add_trace(
                go.Scatter(
                    x=session_data['Session_Index'],
                    y=session_data['max_bpm'],
                    mode='lines+markers',
                    name="Max BPM",
                    marker=dict(color='green')
                ),
                row=row, col=1
            )
            row += 1

        if show_avg_session_duration:
            fig.add_trace(
                go.Scatter(
                    x=session_data['Session_Index'],
                    y=session_data['avg_session_duration'],
                    mode='lines+markers',
                    name="Average Session Duration (hours)",
                    marker=dict(color='purple')
                ),
                row=row, col=1
            )

        # Update layout
        fig.update_layout(
            title_text="Time Series Analysis by Session Index",
            template='plotly_dark',
            plot_bgcolor='black',
            paper_bgcolor='black',
            height=300 * row,  # Adjust height based on visible rows
            showlegend=True,
            xaxis_title="Session Index",
            yaxis_title="Values"
        )

        # Display the figure
        st.plotly_chart(fig)
        st.sidebar.header("Select Visualizations")

        # Sidebar options
        show_jointplot = st.sidebar.checkbox("Joint Plot: Weight vs Calories Burned")
        show_histograms = st.sidebar.checkbox("Histograms for All Features")
        show_grouped_plots = st.sidebar.checkbox("Grouped Plots: Weight vs Other Features")
        show_correlation_heatmap = st.sidebar.checkbox("Correlation Heatmap")
        show_boxplots = st.sidebar.checkbox("Box Plots for Outlier Detection")

        # Visualization: Joint Plot
        if show_jointplot:
            st.subheader("Joint Plot: Weight vs Calories Burned")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.jointplot(
                x='Weight (kg)', y='Calories_Burned', data=data, kind='scatter', color='orange', height=8
            )
            plt.suptitle("Joint Plot of Calories Burned vs Weight", fontsize=16, fontweight='bold', y=1.02)
            st.pyplot(plt)
            plt.close()

        # Visualization: Histograms
        if show_histograms:
            st.subheader("Histograms for All Features")
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            fig, axes = plt.subplots(nrows=len(numeric_columns) // 4 + 1, ncols=4, figsize=(20, 20))
            axes = axes.flatten()

            for i, col in enumerate(numeric_columns):
                sns.histplot(x=data[col], ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
                axes[i].tick_params(axis='x', rotation=45)

            # Remove unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)

        # Visualization: Grouped Plots
        if show_grouped_plots:
            st.subheader("Grouped Plots: Weight vs Other Features")
            grouped_data = data.groupby('Weight (kg)')[
                ['Calories_Burned', 'Session_Duration (hours)', 'Fat_Percentage', 'BMI']
            ].mean().reset_index()
            columns_to_plot = ['Calories_Burned', 'Session_Duration (hours)', 'Fat_Percentage', 'BMI']
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
            axes = axes.flatten()

            for i, column in enumerate(columns_to_plot):
                axes[i].plot(grouped_data['Weight (kg)'], grouped_data[column], marker='o', linestyle='-')
                axes[i].set_title(f'{column} vs Weight (kg)', fontsize=12)
                axes[i].set_xlabel('Weight (kg)', fontsize=10)
                axes[i].set_ylabel(column, fontsize=10)
                axes[i].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

        # Visualization: Correlation Heatmap
        if show_correlation_heatmap:
            st.subheader("Correlation Heatmap")
            correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlation Heatmap')
            st.pyplot(fig)

        # Visualization: Box Plots
        if show_boxplots:
            st.subheader("Box Plots for Outlier Detection")
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            fig, axes = plt.subplots(len(numeric_columns) // 3 + 1, 3, figsize=(15, 15))
            axes = axes.flatten()

            for i, column in enumerate(numeric_columns):
                sns.boxplot(y=data[column], ax=axes[i], color='lightcoral')
                axes[i].set_title(f'Boxplot of {column}', fontsize=12)
                axes[i].set_xlabel(column, fontsize=10)

            # Remove unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)
with tabs[2]:
        st.title("BMI Prediction and Analysis App")

        st.sidebar.header("User Input Features")

        def user_input_features():
            height = st.sidebar.number_input("Height (in meters)", min_value=0.5, max_value=2.5, value=1.80, step=0.01)
            weight = st.sidebar.number_input("Weight (in kilograms)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25, step=1)
            gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
            workout_type = st.sidebar.selectbox("Workout Type", ("Cardio", "Strength", "Mixed"))
            
            data = {
                'Height (m)': height,
                'Weight (kg)': weight,
                'Age': age,
                'Gender': gender,
                'Workout_Type': workout_type
            }
            return pd.DataFrame(data, index=[0])

        # Generate User Input Data
        data = user_input_features()

        # Simulated dataset (replace this with your actual data)
        np.random.seed(42)
        simulated_data = {
            'Height (m)': np.random.uniform(1.5, 2.0, 100),
            'Weight (kg)': np.random.uniform(50, 100, 100),
            'Age': np.random.randint(18, 60, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Workout_Type': np.random.choice(['Cardio', 'Strength', 'Mixed'], 100),
            'BMI': np.random.uniform(18, 30, 100)
        }
        simulated_df = pd.DataFrame(simulated_data)

        # Encode categorical features
        label_encoders = {
            'Gender': LabelEncoder(),
            'Workout_Type': LabelEncoder()
        }
        for col in ['Gender', 'Workout_Type']:
            simulated_df[col] = label_encoders[col].fit_transform(simulated_df[col])

        # Encode user input
        for col in ['Gender', 'Workout_Type']:
            if col in data:
                data[col] = data[col].apply(
                    lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
                )

        # Features and target
        X = simulated_df.drop(columns=['BMI'])
        y = simulated_df['BMI']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X_scaled)

        # Train models
        models = {
            'KNN': KNeighborsRegressor(n_neighbors=10, metric='minkowski', p=2),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        model_results = {}
        for name, model in models.items():
            model.fit(X_pca, y)
            model_results[name] = model

        # Scale and transform user input
        user_scaled = scaler.transform(data)
        user_pca = pca.transform(user_scaled)

        # Allow user to select the model
        selected_model_name = st.sidebar.selectbox(
            "Select a Model for BMI Prediction:",
            list(models.keys())  # Display all model names
        )

        # Retrieve the selected model
        selected_model = model_results[selected_model_name]

        # Display Predictions using the selected model
        bmi_prediction = selected_model.predict(user_pca)[0]
        st.subheader(f"Predicted BMI using {selected_model_name}:")
        st.write(f"{selected_model_name} Model Prediction: {bmi_prediction:.2f}")

        # Evaluate Models
        model_evaluation = []

        st.subheader("Model Evaluation:")
        X_train_pca, X_test_pca, y_train, y_test = X_pca[:80], X_pca[80:], y[:80], y[80:]
        for name, model in model_results.items():
            y_pred = model.predict(X_test_pca)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            model_evaluation.append([name, mae, r2])

        # Create a DataFrame for the evaluation results
        evaluation_df = pd.DataFrame(model_evaluation, columns=["Model", "MAE", "R²"])

        # Display the evaluation table in Streamlit
        st.table(evaluation_df)  

        # Plot Predictions for selected model
        st.subheader(f"Actual vs Predicted BMI ({selected_model_name}):")
        y_pred = selected_model.predict(X_test_pca)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predictions')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Ideal Prediction')
        ax.set_xlabel('Actual BMI')
        ax.set_ylabel('Predicted BMI')
        ax.set_title(f'Actual vs Predicted BMI ({selected_model_name})')
        ax.legend()
        st.pyplot(fig)