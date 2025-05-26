<h1 align="center"> Enhanced Analysis of User Perceptions through NLP Approaches  </h1>

### 💻 _Project Description_

This repository contains the code related to the article *"Enhanced Analysis of User Perceptions through Natural Language Processing Approaches"*, in which we propose enhancing the analysis of the System Usability Scale (SUS) by incorporating a textual field, which is mined using natural language processing (NLP) techniques.

### 📁 _Running the project_


The Python version used was **3.10.12**.
All required dependencies are listed in the `requirements.txt` file. Install them with:
```
bash install.sh
```
Run the code with **Streamlit** using:
```
streamlit run dashboard/main.py
```

###  📄 _Input data_

As input, you must provide a `.csv` file containing 12 columns. Each row should represent a single evaluation. The first column must contain an anonymized and unique user identifier. The following 10 columns should correspond to the SUS statements, already formatted on a Likert scale. The final column should contain optional textual comments provided by users.

In addition, you must provide a `.txt` file containing sample comments along with their sentiment analysis labels.

### ⚙️ _NLP Modules_

This project includes three NLP modules:
1. **Topic Modeling**  
   Groups user comments into semantically coherent topics.
2. **Sentiment Analysis**  
   Determines the sentiment of each comment based on the classes defined in the input `.txt` file.
3. **Summarization**  
   Automatically generates summaries of the comments within each topic.

### 💭 _Citation_

If you used this research or code in your work, please cite it.
```bibtex
@article{TODO,
  author = {TODO},
  title = {TODO},
  journal = {TODO},
  year = {TODO}
}