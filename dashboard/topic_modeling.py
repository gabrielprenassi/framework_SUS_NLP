import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import multiple_choice_answers
import SUS
import overview as ov
from overview import load_topic_summary

colors_labels = {
    'criticism': "#ffcccc",
    'positive feedback': "#c8e6c9",
    'suggestion': "#F0E68C",
    'not pertinent': "#D3D3D3"
}

def plot_comment_distribution(results_df):
    categories = {
        "criticism": "Criticism",
        "positive feedback": "Positive Feedback",
        "suggestion": "Suggestion",
        "not pertinent": "Not Pertinent"
    }

    colors = list(colors_labels.values())
    comment_counts = results_df["results"].value_counts().to_dict()
    
    labels = [categories.get(key, key) for key in categories.keys()]
    values = [comment_counts.get(key, 0) for key in categories.keys()]
    total_comments = sum(values)
    percentages = [(count / total_comments) * 100 if total_comments > 0 else 0 for count in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in percentages],
        textposition="outside"
    ))
    fig.update_layout(
        title="Distribution of Comment Types",
        xaxis_title="Number of Comments",
        yaxis_title="Category",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

def docs_by_word(labels, df_topic_modeling, topic_number):
    results_df = pd.read_csv('./data/results.csv', encoding="utf-8")
    comments_df = pd.read_csv('data/dataFrame.csv')

    relevant_ids = df_topic_modeling[df_topic_modeling['dominant_topic'] == int(topic_number)]['document_id'].tolist()
    filtered_comments = comments_df[comments_df['ID'].isin(relevant_ids)].copy()
    filtered_results = results_df[results_df["ID"].isin(relevant_ids)].copy()
    
    merged_df = filtered_comments.merge(filtered_results, on="ID", how="left", suffixes=('_comments', ''))
    results_df = merged_df.drop(columns=['comments_comments', 'sus_comments'])
    
    word = st.selectbox(
        "Choose 'View all comments' or select a topic word to filter comments:", 
        ['View all comments'] + labels,
        key='aba2'
    )
    if word != 'View all comments':
        results_df = results_df[results_df['clean_text'].str.contains(word, case=False, na=False)]

    type_of_comment = st.selectbox(
        "Choose the type of comment to view:", 
        ['All Comments', 'Only Positive Feedback', 'Only Criticism', 'Only Not Pertinent', 'Only Suggestions'],
        key='aba5'
    )

    st.markdown("""
        <style>
            div[data-testid="stSlider"] {
                width: 100% !important;
                margin: right;
            }
        </style>
    """, unsafe_allow_html=True)

    min_sus, max_sus = st.slider(
        "Select the SUS metric range:",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=1
    )
    results_df = results_df[(results_df['sus'] >= min_sus) & (results_df['sus'] <= max_sus)]
    participants_total = len(results_df)

    if type_of_comment == 'All Comments':
        plot_comment_distribution(results_df)
    elif type_of_comment == 'Only Positive Feedback':
        results_df = results_df[results_df['results'] == 'positive feedback']
    elif type_of_comment == 'Only Criticism':
        results_df = results_df[results_df['results'] == 'criticism']
    elif type_of_comment == 'Only Not Pertinent':
        results_df = results_df[results_df['results'] == 'not pertinent']
    elif type_of_comment == 'Only Suggestions':
        results_df = results_df[results_df['results'] == 'suggestion']

    st.markdown(
        f"<div style='text-align: right;'><strong>Total Users:</strong> {len(results_df)} "
        f"({0 if participants_total == 0 else (len(results_df)/participants_total) * 100:.2f}%)</div>",
        unsafe_allow_html=True
    )

    for _, d in results_df.iterrows():
        text = d['comments']
        label = d['results']
        sus = d['sus']
        ID = d['ID']
        color = colors_labels.get(label, "#FFFFFF")
        st.markdown(
            f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
            f"<strong>User {ID} (SUS: {sus})</strong><br>{text}</div>",
            unsafe_allow_html=True
        )
    
def get_topic_summary(topic_number, topic_amount):
    topic_summary_file = f'summarization/outLLM/detailed_summarization/{topic_amount}/summary_topic_{int(topic_number)}.txt'
    topic_summary = load_topic_summary(topic_summary_file)
    st.markdown(
        f"<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        f"\n\n{topic_summary}"
        f"</div>",
        unsafe_allow_html=True
    )

@st.cache_data
def load_data(path_topic_model, path_topic_modeling, data_sentiment_path):
    return (
        json.load(open(path_topic_model, 'r')),
        pd.read_csv(path_topic_modeling),
        pd.read_csv(data_sentiment_path)
    )
            
def render(topic_number, topic_amount):
    topics_model, df_topic_modeling, df_data = load_data(
        path_topic_model=f'topic_modeling/data_num_topics/{topic_amount}/topics_{topic_amount}.json',
        path_topic_modeling=f'topic_modeling/data_num_topics/{topic_amount}/documents_scores.csv',
        data_sentiment_path='data/dataFrame.csv'
    )

    td_sorted = df_topic_modeling[
        df_topic_modeling['dominant_topic'] == int(topic_number)
    ].sort_values(by='document_score', ascending=False)
    ids = np.array(td_sorted['document_id'].tolist())
    df_data = df_data[df_data['ID'].isin(ids)]

    df_clean_text = pd.read_csv("data/results.csv")
    df_data = df_data.merge(df_clean_text[['ID', 'clean_text']], how='left', on='ID')

    word_and_importance = topics_model[topic_number]
    sorted_word_and_importance = sorted(word_and_importance, key=lambda x: x[1])
    labels, _ = zip(*sorted_word_and_importance)
    labels = list(labels)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Topic Analysis",
        "Topic Summarization",
        "Comments Analysis",
        "Agreement/Disagreement Statements",
        "Topic SUS Metric"
    ])
    with tab1:
        ov.render_specific_topic(int(topic_number), topic_amount)
    with tab2:
        get_topic_summary(topic_number, topic_amount)
    with tab3:
        docs_by_word(labels[::-1], df_topic_modeling, topic_number)
    with tab4:
        multiple_choice_answers.render(
            df_data.drop(columns=['comments']),
            topic_modeling=True,
            labels=labels[::-1]
        )
    with tab5:
        st.title(f"SUS Metric Analysis for Participants of Topic {int(topic_number) + 1}")
        SUS.render(df_data, topic_modeling=True, labels=labels[::-1])
