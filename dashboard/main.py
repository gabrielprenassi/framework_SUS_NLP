import streamlit as st
import os
import sys
import pandas as pd
import multiple_choice_answers
import SUS
import topic_modeling
import overview
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocessing import preprocess_text_pipeline
from topicmodeling.topicmodeling import run_topic_modeling
from sentiment_analysis.run_llms_classifiication import run_classification

st.set_page_config(page_title="MGI - Prototype", layout="wide", initial_sidebar_state="expanded")
CSV_UPLOAD_FOLDER = "data"
TXT_UPLOAD_FOLDER = "txt_data"
os.makedirs(CSV_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TXT_UPLOAD_FOLDER, exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def csv_upload_page():
    st.title("Upload CSV")
    st.selectbox("Select language:", ["Portuguese", "English [Default]"])
    st.subheader("Upload your CSV file")
    uploaded_csv = st.file_uploader("Select a CSV file", type=["csv"], key="csv_file")
    success = False

    if uploaded_csv is not None:
        try:
            header = pd.read_csv(uploaded_csv, nrows=0).columns.tolist()
            if len(header) != 12:
                st.error("The file must contain exactly 12 columns.")
            else:
                dtypes = {header[11]: str}
                uploaded_csv.seek(0)
                df = pd.read_csv(uploaded_csv, dtype=dtypes)
                valid_numbers = True
                for col in df.columns[:11]:
                    try:
                        pd.to_numeric(df[col])
                    except Exception:
                        valid_numbers = False
                        break
                if not valid_numbers:
                    st.error("The first 11 columns must contain only numeric values.")
                else:
                    last_col = df.columns[11]
                    def is_numeric_string(x):
                        x_str = str(x).strip()
                        try:
                            float(x_str)
                            return not any(c.isalpha() for c in x_str)
                        except:
                            return False
                    if df[last_col].dropna().apply(lambda x: not is_numeric_string(x)).all():
                        st.success("CSV accepted!")
                        save_path = os.path.join(CSV_UPLOAD_FOLDER, "dataFrame.csv")
                        with open(save_path, "wb") as f:
                            f.write(uploaded_csv.getbuffer())
                        st.info(f"File saved at: `{save_path}`")
                        success = True
                    else:
                        st.error("The last column must contain only string values.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.subheader("Upload your TXT file with stopwords")
    txt_stopwords = st.file_uploader("Select a TXT file", type=["txt"], key="txt_stopwords")
    if txt_stopwords is not None:
        try:
            text_content = txt_stopwords.read().decode("utf-8")
            st.text_area("Text file content:", text_content, height=200)
            txt_save_path = os.path.join(TXT_UPLOAD_FOLDER, "stopwords.txt")
            with open(txt_save_path, "wb") as f:
                f.write(txt_stopwords.getbuffer())
            st.info(f"Text file saved at: `{txt_save_path}`")
            st.session_state["stopwords_uploaded"] = True
        except Exception as e:
            st.error(f"Error processing text file: {e}")

    st.subheader("Upload your JSON file with the sentiment analysis")
    json_sentiment_analysis = st.file_uploader("Select a JSON file", type=["json"], key="json_sentiment_analysis")

    if json_sentiment_analysis is not None:
        try:
            json_content = json_sentiment_analysis.read().decode("utf-8")
            sentiment_data = json.loads(json_content)
            st.json(sentiment_data)  

            from collections import Counter
            label_counts = Counter(sentiment_data.values())
            
            st.write("Label counts:", label_counts)

            min_count = min(label_counts.values())
            
            st.write(f"Minimum number of examples among labels: {min_count}")

            selected_quantity = st.selectbox(
                "Select number of sentences per sentiment to use:",
                options=list(range(1, min_count + 1)),
                index=min_count - 1  
            )

            st.session_state["selected_quantity_per_label"] = selected_quantity

            json_save_path = os.path.join(TXT_UPLOAD_FOLDER, "sentiment_analysis.json")
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(sentiment_data, f, ensure_ascii=False, indent=4)

            st.info(f"JSON file saved at: `{json_save_path}`")

            st.session_state["sentiment_uploaded"] = True

        except Exception as e:
            st.error(f"Error processing JSON file: {e}")

        return success

def pre_processing_df():
    df = load_data("data/dataFrame.csv")
    df['X'] = df.iloc[:, [1, 3, 5, 7, 9]].sum(axis=1) - 5
    df['Y'] = 25 - df.iloc[:, [2, 4, 6, 8, 10]].sum(axis=1)
    df['sus'] = df.iloc[:, [12, 13]].sum(axis=1) * 2.5
    df.columns.values[11] = "comments"
    df.to_csv('data/dataFrame.csv', index=False)

def get_topic_title(topic_amount, topic_number):
    file_path = f"summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(topic_number)}.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            title = file.read().replace('"', '').strip()
        return title[:-1] if title.endswith('.') else title
    except FileNotFoundError:
        return f"File not found: {file_path}"

def load_all_topic_titles(topic_amount):
    return [get_topic_title(topic_amount, topic_number) for topic_number in range(topic_amount)]

def main_app():
    selected_quantity = st.session_state.get("selected_quantity_per_label")
    
    st.info("Starting: Preprocessing DataFrame...")
    pre_processing_df()
    st.success("DataFrame preprocessing completed.")

    st.info("Starting: Text Preprocessing Pipeline...")
    preprocess_text_pipeline()
    st.success("Text preprocessing completed.")

    st.info("Starting: Topic Modeling...")
    run_topic_modeling()
    st.success("Topic modeling completed.")

    st.info("Starting: Sentiment Classification...")
    run_classification(number_of_examples=selected_quantity)
    st.success("Classification completed.")

    st.info("Starting: Detailed Summarization...")
    from summarization.detailed_topic_comments_summarization import run_detailed
    run_detailed()
    st.success("Detailed summarization completed.")

    st.info("Starting: Concise Summarization...")
    from summarization.concise_topic_comments_summarization import run_concise
    run_concise()
    st.success("Concise summarization completed.")

    st.info("Starting: Single-sentence Summarization...")
    from summarization.single_sentence_topic_summarization import run_single
    run_single()
    st.success("Single-sentence summarization completed.")

    st.info("Loading DataFrame for results display...")

    df = load_data('data/dataFrame.csv')

    selected_columns = ["ID", "sus", "comments"]
    df_results = df[selected_columns].copy()
    df_results = df_results[df_results["comments"].notna()].reset_index(drop=True)

    with open('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json', "r") as file:
        classification_data = json.load(file)
    y_pred_text = classification_data.get("y_pred_text", [])
    df_results["results"] = y_pred_text[:len(df_results)]

    df_results.to_csv('data/results.csv', index=False)

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Overview", "SUS Analyses", "Topic Modeling"])
    topic_amount = st.sidebar.selectbox("Select Number of Topics", (5, 10, 15))

    if selection == "Overview":
        overview.render_overview(df, topic_amount)
    elif selection == "SUS Analyses":
        tab1, tab2 = st.tabs(["Agreement/Disagreement Statements", "Global SUS"])
        with tab1:
            multiple_choice_answers.render(df)
        with tab2:
            SUS.render(df)
    elif selection == "Topic Modeling":
        topic_titles = load_all_topic_titles(topic_amount)
        selected_topic_title = st.sidebar.selectbox("Select a Topic:", options=topic_titles)
        topic_number = topic_titles.index(selected_topic_title)
        topic_modeling.render(topic_number=str(topic_number), topic_amount=topic_amount)

if "csv_uploaded" not in st.session_state:
    st.session_state["csv_uploaded"] = False
if "sentiment_uploaded" not in st.session_state:
    st.session_state["sentiment_uploaded"] = False
if "stopwords_uploaded" not in st.session_state:
    st.session_state["stopwords_uploaded"] = False
if "proceed_main" not in st.session_state:
    st.session_state["proceed_main"] = False

if not (st.session_state["csv_uploaded"] and st.session_state["sentiment_uploaded"] and st.session_state["stopwords_uploaded"]):
    success = csv_upload_page()
    if success:
        st.session_state["csv_uploaded"] = True
    st.stop()

if st.session_state["proceed_main"]:
    main_app()

if st.session_state["csv_uploaded"] and st.session_state["sentiment_uploaded"] and st.session_state["stopwords_uploaded"]:
    st.subheader("Ready to proceed?")
    proceed = st.button("Proceed to Analysis")
    if proceed:
        st.session_state["proceed_main"] = True
        st.rerun() 
