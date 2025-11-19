import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# -------------------- LOAD ENVIRONMENT VARIABLES --------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# -------------------- STREAMLIT SETUP --------------------
st.set_page_config(page_title="AI Data Analyst Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Data Analyst Agent using CrewAI + Groq")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# -------------------- LEVEL 2: AUTO MODEL SELECTION --------------------
def auto_select_model(X_train, y_train, X_test, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor()
    }
    best_model, best_r2, best_name = None, -999, ""
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name
    return best_model, best_r2, best_name

# -------------------- MODEL TRAINING (with fix) --------------------
def train_model(data):
    # Drop non-numeric and ID-like columns
    data = data.loc[:, ~data.columns.str.contains("id|code|name", case=False)]
    data = data.select_dtypes(include="number")

    numeric_cols = data.columns
    if len(numeric_cols) < 2:
        return None, None, None, "Not enough numeric columns to train a model."

    X = data[numeric_cols[:-1]]
    y = data[numeric_cols[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model, best_r2, best_name = auto_select_model(X_train, y_train, X_test, y_test)
    preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return best_model, best_r2, mse, f"{best_name} selected | R²: {best_r2:.3f}, MSE: {mse:.3f}"

# -------------------- AGENT LOGIC (LEVEL 1) --------------------
def run_ai_agents(data_path, llm):
    data = pd.read_csv(data_path)

    data_loader = Agent(
        role="Data Loader",
        goal="Load and understand datasets",
        backstory="You are skilled at exploring and describing datasets for analysis.",
        llm=llm,
        verbose=False,
    )

    insight_agent = Agent(
        role="Insight Generator",
        goal="Analyze dataset quality and provide actionable insights",
        backstory="You detect patterns, correlations, and anomalies in data for better modeling.",
        llm=llm,
        verbose=False,
    )

    model_trainer = Agent(
        role="Model Trainer",
        goal="Train a regression model and summarize its performance",
        backstory="You build and test regression models on provided data.",
        llm=llm,
        verbose=False,
    )

    advisor = Agent(
        role="Advisor",
        goal="Analyze model results and provide improvement suggestions",
        backstory="You interpret ML metrics and suggest better approaches.",
        llm=llm,
        verbose=False,
    )

    # Tasks
    load_task = Task(
        description=f"Load and summarize the dataset located at {data_path}",
        agent=data_loader,
        expected_output="A summary of dataset columns, data types, and basic statistics."
    )

    insight_task = Task(
        description=f"Perform data quality checks and correlation analysis for {data_path}",
        agent=insight_agent,
        expected_output="Insights about outliers, missing values, and feature relationships."
    )

    model, r2, mse, msg = train_model(data)
    train_task = Task(
        description=f"Train model completed with R²={r2:.3f}, MSE={mse:.3f}. Summarize the model performance.",
        agent=model_trainer,
        expected_output="A concise summary of model training results and evaluation metrics."
    )

    evaluate_task = Task(
        description=f"Interpret results and give suggestions to improve model accuracy. R²={r2:.3f}, MSE={mse:.3f}",
        agent=advisor,
        expected_output="A list of insights and recommendations to improve the model."
    )

    crew = Crew(
        agents=[data_loader, insight_agent, model_trainer, advisor],
        tasks=[load_task, insight_task, train_task, evaluate_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff(inputs={"data_path": data_path})
    return result, r2, mse, model

# -------------------- MAIN APP --------------------
if uploaded_file:
    data_path = "uploaded_data.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    data = pd.read_csv(data_path)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    if st.button("Run AI Analysis"):
        with st.spinner("🤖 Agents are working... please wait..."):
            try:
                llm = LLM(
                    model="groq/llama-3.1-8b-instant",
                    api_key=api_key,
                    temperature=0.4
                )

                result, r2, mse, model = run_ai_agents(data_path, llm)

                st.success("✅ Analysis Complete")
                st.subheader("📈 Model Performance")
                st.write(f"R² Score: {r2:.3f}")
                st.write(f"MSE: {mse:.3f}")

                st.subheader("🧠 AI Agent Summary")
                if hasattr(result, "raw") and result.raw:
                    st.markdown(result.raw)
                else:
                    st.write(result)

                # -------------------- LEVEL 3: VISUAL REASONING DASHBOARD --------------------
                st.subheader("📉 Visual Reasoning Dashboard")

                # Correlation Heatmap
                st.markdown("**Correlation Heatmap**")
                corr = data.select_dtypes(include="number").corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True)
                st.pyplot(fig)

                # Target distribution
                st.markdown("**Target Variable Distribution**")
                fig2, ax2 = plt.subplots()
                data[data.select_dtypes(include="number").columns[-1]].hist(ax=ax2, bins=20, color="skyblue", edgecolor="black")
                ax2.set_title("Target Variable Distribution")
                st.pyplot(fig2)

                # Residual plot (if applicable)
                if isinstance(model, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor)):
                    X = data.select_dtypes(include="number").iloc[:, :-1]
                    y = data.select_dtypes(include="number").iloc[:, -1]
                    preds = model.predict(X)
                    residuals = y - preds
                    st.markdown("**Residual Plot**")
                    fig3, ax3 = plt.subplots()
                    sns.scatterplot(x=preds, y=residuals, ax=ax3)
                    ax3.axhline(0, color='red', linestyle='--')
                    ax3.set_xlabel("Predicted Values")
                    ax3.set_ylabel("Residuals")
                    st.pyplot(fig3)

            except Exception as e:
                st.error(f"⚠️ Error occurred: {e}")
