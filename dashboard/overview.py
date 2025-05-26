import streamlit as st
import plotly.express as px
import pandas as pd
import json
import plotly.graph_objects as go

def load_topic_summary(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def create_card_with_score(question, score, background_color):
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 10px 15px; margin-bottom: 10px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center;'>"
        f"<span style='font-size: 18px;'><strong>{question}</strong></span>"
        f"<span style='font-size: 24px;'><strong>{score:.2f}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def calculate_means(df):
    numeric_columns = df.select_dtypes(include=["number"]).columns
    numeric_columns = numeric_columns[1:-3]

    means = {}
    original_means = {}

    for idx, column in enumerate(numeric_columns):
        mean = df[column].mean()
        original_means[column] = mean
        if idx % 2 != 0:
            mean = 6 - mean
        means[column] = mean

    max_mean_question = max(means, key=means.get)
    min_mean_question = min(means, key=means.get)

    return (
        means,
        (max_mean_question, means[max_mean_question]),
        (min_mean_question, means[min_mean_question]),
        original_means,
    )

def calculate_sentiment_totals(df_topic_modeling, topic_amount):
    df_results = load_data('data/results.csv')
    df_topic_modeling = load_data(f'topic_modeling/data_num_topics/{topic_amount}/documents_scores.csv')

    class_count_df = count_classes_per_topic(df_results, df_topic_modeling)
    class_count_df["total"] = (
        class_count_df["criticism"]
        + class_count_df["positive feedback"]
        + class_count_df["suggestion"]
        + class_count_df["not pertinent"]
    )
    class_count_df['positive_feedback_rate'] = (
        class_count_df['positive feedback'] / class_count_df['total']
    ) * 100
    class_count_df['criticism_rate'] = (
        class_count_df['criticism'] / class_count_df['total']
    ) * 100

    most_positive_topic = class_count_df['positive_feedback_rate'].idxmax()
    most_critical_topic = class_count_df['criticism_rate'].idxmax()

    return class_count_df, most_positive_topic, most_critical_topic

def create_card(content, background_color):
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>"
        f"{content}"
        f"</div>",
        unsafe_allow_html=True,
    )

def create_percentage_bar_chart(positive_feedbacks, criticisms, suggestions, not_pertinent, title=""):
    total = positive_feedbacks + criticisms + suggestions + not_pertinent
    positive_rate = round((positive_feedbacks / total) * 100, 1) if total > 0 else 0.0
    criticism_rate = round((criticisms / total) * 100, 1) if total > 0 else 0.0
    suggestion_rate = round((suggestions / total) * 100, 1) if total > 0 else 0.0
    not_pertinent_rate = round((not_pertinent / total) * 100, 1) if total > 0 else 0.0

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Comments"],
        x=[criticism_rate],
        orientation='h',
        marker=dict(color="#FFA6B1"),
        name="Criticism",
        text=f"{criticism_rate:.2f}%",
        textposition='inside'
    ))
    fig.add_trace(go.Bar(
        y=["Comments"],
        x=[suggestion_rate],
        orientation='h',
        marker=dict(color="#F0E68C"),
        name="Suggestion",
        text=f"{suggestion_rate:.2f}%",
        textposition='inside'
    ))
    fig.add_trace(go.Bar(
        y=["Comments"],
        x=[not_pertinent_rate],
        orientation='h',
        marker=dict(color="#D3D3D3"),
        name="Not Pertinent",
        text=f"{not_pertinent_rate:.2f}%",
        textposition='inside'
    ))
    fig.add_trace(go.Bar(
        y=["Comments"],
        x=[positive_rate],
        orientation='h',
        marker=dict(color="#86E886"),
        name="Positive Feedback",
        text=f"{positive_rate:.2f}%",
        textposition='inside'
    ))

    top_margin = 50 if title else 10
    height = 200 if title else 150

    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            title="Percentage (%)",
            range=[0, 100],
            ticksuffix='%'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=14)
        ),
        plot_bgcolor="white",
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=top_margin, b=10),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        )
    )

    return fig

def render_topic_words(topic_number, topic_amount, x=0, title=""):
    try:
        with open(f'topic_modeling/data_num_topics/{topic_amount}/topics_{topic_amount}.json', 'r') as file:
            topics_model = json.load(file)
    except Exception as e:
        st.error(f"Error loading topic data: {e}")
        return

    words_importance = topics_model.get(str(topic_number), [])
    valid_words_importance = [item for item in words_importance if isinstance(item, list) and len(item) == 2]
    if len(valid_words_importance) == 0:
        st.warning(f"Topic {topic_number} has no relevant words.")
        return

    valid_words_importance = sorted(valid_words_importance, key=lambda x: x[1], reverse=True)
    labels, values = zip(*valid_words_importance)

    fig = go.Figure(go.Bar(
        y=list(reversed(labels)),
        x=list(reversed(values)),
        orientation='h',
        marker=dict(color='#3BCBDE')
    ))

    top_margin = 10

    if(title != ""):
        top_margin = 50

    fig.update_layout(
        title=title,
        xaxis_title="Relevance",
        yaxis_title="Words",
        plot_bgcolor="white",
        height=300,
        margin=dict(l=10, r=10, t=top_margin, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_words_chart_{topic_number}_{x}")

def count_classes_per_topic(df_results, df_documents_scores):
    df_merged = df_results.merge(
        df_documents_scores,
        left_on="ID",
        right_on="document_id",
        how="inner"
    )
    class_count_per_topic = df_merged.groupby("dominant_topic")["results"].value_counts().unstack(fill_value=0)
    return class_count_per_topic

def render_overview_topics(topic_amount):
    col1, col2 = st.columns([3, 1])

    with col2:
        sorting_option = st.selectbox(
            " ",
            options=["Most Positive Feedback", "Most Criticism"],
            index=0,
            label_visibility="collapsed"
        )

    df_results = load_data('data/results.csv')
    df_topic_modeling = load_data(f'topic_modeling/data_num_topics/{topic_amount}/documents_scores.csv')
    class_count_df = count_classes_per_topic(df_results, df_topic_modeling)
    grouped, _, _ = calculate_sentiment_totals(df_topic_modeling, topic_amount)

    if sorting_option == "Most Positive Feedback":
        grouped = grouped.sort_values(by='positive_feedback_rate', ascending=False)
    elif sorting_option == "Most Criticism":
        grouped = grouped.sort_values(by='criticism_rate', ascending=False)
        
    for topic_number, _ in grouped.iterrows():
        topic_title_file = f'summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(topic_number)}.txt'
        topic_title = load_topic_summary(topic_title_file).replace('"', '').replace('.', '')
        st.markdown(f"#### {topic_title}")

        col1, col2 = st.columns(2, gap="medium")

        positive_feedbacks = class_count_df['positive feedback'][topic_number]
        criticisms = class_count_df['criticism'][topic_number]
        suggestions = class_count_df['suggestion'][topic_number]
        not_pertinent = class_count_df['not pertinent'][topic_number]
        total_comments = positive_feedbacks + criticisms + suggestions + not_pertinent

        with col1:
            render_topic_words(topic_number, topic_amount, title="Relevance of Topic Words")
            st.markdown(
                "<h6 style='font-weight: 900; margin-top: 10px;'>Comments in Topic</h6>",
                unsafe_allow_html=True
            )
            create_card(
                content=f"{total_comments} Comments",
                background_color="#f8f9fa"
            )

        with col2:
            summary_file = f'summarization/outLLM/concise_summarization/{topic_amount}/summary_topic_{int(topic_number)}.txt'
            try:
                topic_summary = load_topic_summary(summary_file)
            except:
                topic_summary = "No summary available for this topic."

            st.markdown(
                "<h6 style='font-weight: 900; margin-top: 10px;'>Topic Summary</h6>",
                unsafe_allow_html=True
            )
            create_card(
                content=f"{topic_summary}",
                background_color="#f8f9fa"
            )

            positive_feedbacks = class_count_df['positive feedback'][topic_number]
            criticisms = class_count_df['criticism'][topic_number]
            suggestions = class_count_df['suggestion'][topic_number]
            not_pertinent = class_count_df['not pertinent'][topic_number]

            fig = create_percentage_bar_chart(
                positive_feedbacks,
                criticisms,
                suggestions,
                not_pertinent,
                "Percentage of Comments by Sentiment"
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_chart_{int(topic_number)}")

        st.markdown("---")

def render_specific_topic(topic_number, topic_amount):
    topic_title_file = f'summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(topic_number)}.txt'
    topic_title = load_topic_summary(topic_title_file).replace('"', '').replace('.', '')
    st.markdown(f"#### {topic_title}")

    summary_file = f'summarization/outLLM/concise_summarization/{topic_amount}/summary_topic_{int(topic_number)}.txt'
    try:
        topic_summary = load_topic_summary(summary_file)
    except:
        topic_summary = "No summary available for this topic."

    df_results = load_data('data/results.csv')
    df_topic_modeling = load_data(f'topic_modeling/data_num_topics/{topic_amount}/documents_scores.csv')
    class_count_df = count_classes_per_topic(df_results, df_topic_modeling)
    grouped, _, _ = calculate_sentiment_totals(df_topic_modeling, topic_amount)

    if topic_number in grouped.index:
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            render_topic_words(topic_number, topic_amount, title="Relevance of Topic Words")

        with col2:
            st.markdown(
                "<h6 style='font-weight: 900; margin-top: 10px;'>Topic Summary</h6>",
                unsafe_allow_html=True
            )
            create_card(
                content=f"{topic_summary}",
                background_color="#f8f9fa"
            )

            positive_feedbacks = class_count_df['positive feedback'][topic_number]
            criticisms = class_count_df['criticism'][topic_number]
            suggestions = class_count_df['suggestion'][topic_number]
            not_pertinent = class_count_df['not pertinent'][topic_number]

            fig = create_percentage_bar_chart(
                positive_feedbacks,
                criticisms,
                suggestions,
                not_pertinent,
                "Percentage of Comments by Sentiment"
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_chart_{int(topic_number)}")
    else:
        st.warning(f"Topic {topic_number + 1} not found.")

def render_response_percentages(df, question):
    color_mapping = {
        "Strongly Agree": '#1a9850',
        "Somewhat Agree": '#98df8a',
        "Neither Agree nor Disagree": '#fee08b',
        "Somewhat Disagree": '#fc8d59',
        "Strongly Disagree": '#d73027'
    }

    response_labels = {
        1: "Strongly Disagree",
        2: "Somewhat Disagree",
        3: "Neither Agree nor Disagree",
        4: "Somewhat Agree",
        5: "Strongly Agree"
    }

    response_counts = df[question].dropna().value_counts()
    total = response_counts.sum()
    response_counts.index = response_counts.index.map(response_labels)

    st.markdown(
        """
        <style>
        .response-box {
            background-color: #f0f0f0;
            padding: 15px;
            margin-top: 5px;
            border-radius: 5px;
        }
        .response-box ul {
            list-style-type: none;
            padding-left: 0;
        }
        .response-box li {
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }
        .color-square {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 8px;
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    list_items = ""
    for response, count in response_counts.items():
        percentage = (count / total) * 100
        color = color_mapping.get(response, "#000000")
        list_items += (
            f"<li>"
            f"<span class='color-square' style='background-color:{color};'></span>"
            f"<strong>{response}:</strong> {count} responses ({percentage:.2f}%)"
            f"</li>"
        )

    st.markdown(
        f"""
        <div class="response-box">
          <h4>Distribution of Responses:</h4>
          <ul>
            {list_items}
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_questions_analysis(df, max_item, min_item, original_means):
    best_question = max_item[0][2:].strip()
    best_score = original_means[max_item[0]]

    worst_question = min_item[0][2:].strip()
    worst_score = original_means[min_item[0]]

    st.markdown("#### Question with Highest Score")
    create_card_with_score(
        question=best_question,
        score=best_score,
        background_color="#86E886"
    )
    render_response_percentages(df, max_item[0])

    st.markdown("---")

    st.markdown("#### Question with Lowest Score")
    create_card_with_score(
        question=worst_question,
        score=worst_score,
        background_color="#FFA6B1"
    )
    render_response_percentages(df, min_item[0])

def render_topic_analysis(most_negative_topic, most_positive_topic, grouped, topic_amount):
    st.markdown("### Analysis of Topic with Highest Positive Feedback Percentage")

    col1, col2 = st.columns(2, gap="large")
    positive_summary = load_topic_summary(
        f"summarization/outLLM/concise_summarization/{topic_amount}/summary_topic_{int(most_positive_topic)}.txt"
    )

    df_results = load_data('data/results.csv')
    df_topic_modeling = load_data(f'topic_modeling/data_num_topics/{topic_amount}/documents_scores.csv')
    class_count_df = count_classes_per_topic(df_results, df_topic_modeling)

    with col1:
        file_path = f"summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(most_positive_topic)}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                title = file.read().replace('"', '').replace('.', '')
        except FileNotFoundError:
            title = f"File not found: {file_path}"

        create_card(
            content=f"Topic: {title}",
            background_color="#86E886"
        )
        create_card(
            content=f"Summary: {positive_summary}",
            background_color="#86E886"
        )

        positive_feedbacks = class_count_df['positive feedback'][most_positive_topic]
        criticisms = class_count_df['criticism'][most_positive_topic]
        suggestions = class_count_df['suggestion'][most_positive_topic]
        not_pertinent = class_count_df['not pertinent'][most_positive_topic]
        total_comments = positive_feedbacks + criticisms + suggestions + not_pertinent

        st.markdown(
            "<h6 style='font-weight: 900; margin-top: 10px;'>Comments in Topic</h6>",
            unsafe_allow_html=True
        )
        create_card(
            content=f"{total_comments} Comments",
            background_color="#f8f9fa"
        )

    with col2:
        st.markdown("#### Percentage of Comments by Sentiment")
        fig = create_percentage_bar_chart(
            positive_feedbacks,
            criticisms,
            suggestions,
            not_pertinent
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"positive_chart_{most_positive_topic}")

        st.markdown("#### Relevance of Topic Words")
        render_topic_words(most_positive_topic, topic_amount, 1)

    st.markdown("---")

    st.markdown("### Analysis of Topic with Highest Negative Feedback Percentage")

    col1, col2 = st.columns(2, gap="large")
    negative_summary = load_topic_summary(
        f"summarization/outLLM/concise_summarization/{topic_amount}/summary_topic_{int(most_negative_topic)}.txt"
    )

    with col1:
        file_path = f"summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(most_negative_topic)}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                title = file.read().replace('"', '').replace('.', '')
        except FileNotFoundError:
            title = f"File not found: {file_path}"

        create_card(
            content=f"Topic: {title}",
            background_color="#FFA6B1"
        )
        create_card(
            content=f"Summary: {negative_summary}",
            background_color="#FFA6B1"
        )

        positive_feedbacks = class_count_df['positive feedback'][most_negative_topic]
        criticisms = class_count_df['criticism'][most_negative_topic]
        suggestions = class_count_df['suggestion'][most_negative_topic]
        not_pertinent = class_count_df['not pertinent'][most_negative_topic]
        total_comments = positive_feedbacks + criticisms + suggestions + not_pertinent

        st.markdown(
            "<h6 style='font-weight: 900; margin-top: 10px;'>Comments in Topic</h6>",
            unsafe_allow_html=True
        )
        create_card(
            content=f"{total_comments} Comments",
            background_color="#f8f9fa"
        )

    with col2:
        st.markdown("#### Percentage of Comments by Sentiment")
        fig = create_percentage_bar_chart(
            positive_feedbacks,
            criticisms,
            suggestions,
            not_pertinent
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"negative_chart_{most_negative_topic}")

        st.markdown("#### Relevance of Topic Words")
        render_topic_words(most_negative_topic, topic_amount, 2)

def load_classification_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['y_pred_text']

def render_overview(df, topic_amount):
    _, max_item, min_item, original_means = calculate_means(df)
    df_topic_modeling = load_data(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')
    grouped, most_positive_topic, most_negative_topic = calculate_sentiment_totals(df_topic_modeling, topic_amount)

    st.markdown(
        "<h1 style='text-align: center; font-size: 28px; padding: 0px 10px 10px 10px;'>Overview</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            f"<div style='background-color: whitesmoke; padding: 5px; border-radius: 10px; text-align: center;'>"
            f"<strong>Total Participants:</strong> {len(df)}</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div style='background-color: whitesmoke; padding: 5px; border-radius: 10px; text-align: center;'>"
            f"<strong>Total Comments:</strong> {len(df_topic_modeling)}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "General Topic Analysis",
        "Statement Analysis (Best/Worst)",
        "Topic Analysis (Best/Worst)"
    ])

    with tab1:
        render_overview_topics(topic_amount)

    with tab2:
        render_questions_analysis(df, max_item, min_item, original_means)

    with tab3:
        render_topic_analysis(
            most_negative_topic,
            most_positive_topic,
            grouped,
            topic_amount,
        )

if __name__ == "__main__":
    render_overview()
