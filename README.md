<h1 align="center"> Enhanced Analysis of User Perceptions through NLP Approaches  </h1>

### 💻 _Project Description_

This repository contains the code related to the article *"Enhanced Analysis of User Perceptions through Natural Language Processing Approaches"*, in which we propose enhancing the analysis of the System Usability Scale (SUS) by incorporating a textual field, which is mined using natural language processing (NLP) techniques.

### 📁 _Running the project_


The Python version used was **3.10.12**.
All required dependencies are listed in the `requirements.txt` file. Install them with:
```
bash install.sh
```
Activate the virtual environment using:
```
source venv/bin/activate
```
Run the code with **Streamlit** using:
```
streamlit run dashboard/main.py
```

###  📄 _Input data_

As input, you must provide a `.csv` file containing 12 columns. Each row should represent a single evaluation. The first column must contain an anonymized and unique user identifier. The following 10 columns should correspond to the SUS statements, already formatted on a Likert scale. The final column should contain optional textual comments provided by users.

In addition, you must provide a `.txt` file containing sample comments along with their sentiment analysis labels. You can choose the number of examples for the sentiment analysis prompt. After that, the process will begin.

### ⚙️ _NLP Modules_

This project includes three NLP modules:
1. **Topic Modeling**  
   Groups user comments into semantically coherent topics.
2. **Sentiment Analysis**  
   Determines the sentiment of each comment based on the classes defined in the input `.txt` file.
3. **Summarization**  
   Automatically generates summaries of the comments within each topic.

To use the LLaMA models, you must have an access key and add it to `sentiment_analysis/src/llms/token_id.py`. You can adapt the prompts to fit your specific use case.

### 💭 _Citation_

If you used this research or code in your work, please cite it.
```bibtex
@inproceedings{prenassi2025enhanced,
  title={Enhanced Analysis of User Perceptions Through Natural Language Processing Approaches},
  author={Prenassi, Gabriel and Machado, Ana and Freitas, Davi and Lima, Andr{\'e} and Prates, Raquel O and Landim, Antonio and Rocha, Leonardo and Tuler, Elisa},
  booktitle={IFIP Conference on Human-Computer Interaction},
  pages={453--476},
  year={2025},
  organization={Springer}
}
