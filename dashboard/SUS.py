import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from multiple_choice_answers import print_information

categories_information = {
    'Worst Imaginable': {'min_value': 0, 'max_value': 25, 'color': '#800000'},  # Dark red
    'Bad': {'min_value': 25, 'max_value': 50, 'color': '#d73027'},              # Red
    'Ok': {'min_value': 50, 'max_value': 70, 'color': '#fc8d59'},               # Orange
    'Good': {'min_value': 70, 'max_value': 80, 'color': '#fee08b'},             # Yellow
    'Excellent': {'min_value': 80, 'max_value': 85, 'color': '#98df8a'},        # Light green
    'Best Imaginable': {'min_value': 85, 'max_value': 100, 'color': '#1a9850'}   # Green
}

def create_scale_bar():
    fig = go.Figure()
    for category in categories_information:
        fig.add_trace(go.Scatter(
            x=[categories_information[category]["min_value"], categories_information[category]["max_value"]],
            y=[0, 0],
            mode='lines',
            line=dict(color=categories_information[category]["color"], width=20),
            name=category,
            fill='toself'
        ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            tickvals=[0, 25, 50, 70, 80, 85, 100],
            ticktext=["0", "25", "50", "70", "80", "85", "100"],
            title="Rating Scale",
            fixedrange=True
        ),
        yaxis=dict(
            range=[-1, 1],
            showticklabels=False,
            fixedrange=True
        ),
        showlegend=False,
        height=190,
        plot_bgcolor="white",
        autosize=True,
        dragmode=False
    )
    st.plotly_chart(fig)

def create_pie_chart(categories):
    fig = px.pie(
        names=categories.keys(),
        values=categories.values()
    )
    fig.update_traces(
        marker=dict(colors=[categories_information[category]['color'] for category in categories]),
        hovertemplate='<br>Total Participants: %{value}<extra></extra>'
    )
    st.plotly_chart(fig)

def render(df, topic_modeling=False, labels=[]):
    # Dictionary to store the metric value counts for the users
    categories = {
        "Worst Imaginable": 0,
        "Bad": 0,
        "Ok": 0,
        "Good": 0,
        "Excellent": 0,
        "Best Imaginable": 0
    }

    if topic_modeling:
        word = st.selectbox(
            "Choose 'Consider all words' or select a topic word as a filter:",
            ['Consider all words'] + labels,
            key='aba4'
        )
        if word != 'Consider all words':
            df = df[df['clean_text'].str.contains(word, case=False, na=False)]

    df.loc[:, 'sus'] = df['sus'].astype(str).str.replace(',', '.', regex=False).astype(float)

    categories = {}
    for value in df['sus']:
        if value < 25:
            categories.setdefault("Worst Imaginable", 0)
            categories["Worst Imaginable"] += 1
        elif value < 50:
            categories.setdefault("Bad", 0)
            categories["Bad"] += 1
        elif value < 70:
            categories.setdefault("Ok", 0)
            categories["Ok"] += 1
        elif value < 80:
            categories.setdefault("Good", 0)
            categories["Good"] += 1
        elif value < 85:
            categories.setdefault("Excellent", 0)
            categories["Excellent"] += 1
        else:
            categories.setdefault("Best Imaginable", 0)
            categories["Best Imaginable"] += 1

    print_information(
        number_of_users=len(df),
        mean=df['sus'].mean(),
        std=df['sus'].std()
    )

    create_scale_bar()
    create_pie_chart(categories)
