import streamlit as st

mapping = {
    5: "Strongly Agree",
    4: "Agree Partially",
    3: "Neither Agree Nor Disagree",
    2: "Disagree Partially",
    1: "Strongly Disagree"
}

def split_columns_by_type(df):
    questions_numerical = []
    for column in list(df.columns)[1:-4]:
        if any(str(value) in column for value in range(10)):
            questions_numerical.append(column)
    return questions_numerical


def print_information(number_of_users, mean, std):
    if number_of_users == 1:
        std_text = "<div><strong>Standard Deviation:</strong> No standard deviation available.</div>"
    else:
        std_text = f"<div><strong>Standard Deviation:</strong> {std:.2f}</div>"
    st.markdown(
        f"<div><strong>Total Participants:</strong> {number_of_users}</div>"
        f"<div><strong>Mean Value:</strong> {mean:.2f}</div>"
        f"{std_text}",
        unsafe_allow_html=True
    )

color_mapping = {
    "Strongly Agree": '#1a9850',     # Green
    "Agree Partially": '#98df8a',    # Dark green
    "Neither Agree Nor Disagree": '#fee08b',  # Yellow
    "Disagree Partially": '#fc8d59', # Orange
    "Strongly Disagree": '#d73027'   # Red
}

def create_response_info_box(responses_counts):
    total = responses_counts.sum()
    st.markdown(
        """
        <style>
        .gray-box {
            background-color: #f0f0f0;
            padding: 15px;
            margin-top: 15px;
            border-radius: 5px;
        }
        .gray-box ul {
            list-style-type: none;
            padding-left: 0;
        }
        .gray-box li {
            margin-bottom: 8px;
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
    for response, count in responses_counts.items():
        percentage = (count / total) * 100
        color = color_mapping.get(response, "#000000")
        list_items += (
            f"<li>"
            f"<span class='color-square' style='background-color:{color};'></span>"
            f"<strong>{response}:</strong> {count} responses ({percentage:.2f}%)"
            f"</li>"
        )
    st.markdown(f"""
    <div class="gray-box">
        <h4>Response Distribution:</h4>
        <ul>
            {list_items}
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render(df, topic_modeling=False, labels=[]):
    numeric_questions = split_columns_by_type(df)
    selected_question = st.selectbox("Select a statement:", numeric_questions)
    if topic_modeling:
        word = st.selectbox(
            "Select 'Consider all comments' or choose a topic word as filter:",
            ['Consider all comments'] + labels,
            key='aba3'
        )
        if word != 'Consider all comments':
            df = df[df['clean_text'].str.contains(word, case=False, na=False)]
    print_information(
        number_of_users=len(df),
        mean=df[selected_question].mean(),
        std=df[selected_question].std()
    )
    mapped_responses = df[selected_question].map(mapping)
    response_counts = mapped_responses.value_counts()
    create_response_info_box(response_counts)
