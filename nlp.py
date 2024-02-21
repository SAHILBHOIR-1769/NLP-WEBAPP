import streamlit as st
from time import sleep
from stqdm import stqdm
from transformers import pipeline
import json
import spacy
import spacy_streamlit
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def draw_all(
        key,
        plot=False,
):
    st.write(
        """
        # NLP Web App

        Explore the world of Natural Language Processing with this amazing web app! 

        This App harnesses advanced NLP models, enabling it to perform intricate tasks with textual data.

        ```python
        Key Features:
        1. Summarize Text
        2. Recognize Named Entities
        3. Analyze Sentiments
        4. Question Answering



        ```
        """
    )


with st.sidebar:
    draw_all("sidebar")


def main():
    st.title("NLP Web App")
    menu = ["--Select--", "Text Summarizer", "Named Entity Recognition", "Sentiment Analysis", "Question Answering"]
    choice = st.sidebar.selectbox("Choose What u wanna do !!", menu)

    if choice == "--Select--":

        st.write("""

                 Natural Language Processing Web App – a versatile tool designed to effortlessly handle a wide array of text-related tasks.
        """)

        st.write("""

                 Natural Language Processing, is a branch of Artificial Intelligence that enables computers to 
                 understand, interpret, and generate human language in a way that's meaningful and contextually 
                 relevant. It allows machines to interact with and process text or speech data, opening up a wide 
                 range of applications like chatbots, sentiment analysis, language translation, and more. 
                 Essentially, NLP bridges the gap between human communication and computer understanding.



        """)

        st.write("""

                 Experience the convenience and power of NLP
        """)

        st.image('background_image.jpg')



    elif choice == "Text Summarizer":
        st.subheader("Text Summarization")
        st.write(" Enter the Text you want to summarize !")
        st.write("""1. Preprocessing:
Break text into words or phrases.
Remove non-essential elements.
Normalize text (e.g., lowercase).

2. Feature Extraction:
Convert text to numbers (e.g., TF-IDF, embeddings).
Assess word importance based on context.

3. Sentence Scoring:
Score sentences using word importance.
Consider factors like frequency and position.

4. Sentence Selection:
Pick top-scoring sentences for the summary.""")
        raw_text = st.text_area("Your Text", "Enter Your Text Here")
        num_words = st.number_input("Enter Number of Words in Summary")

        if raw_text != "" and num_words is not None:
            num_words = int(num_words)
            summarizer = pipeline('summarization')
            summary = summarizer(raw_text, min_length=num_words, max_length=100)
            s1 = json.dumps(summary[0])
            d2 = json.loads(s1)
            result_summary = d2['summary_text']
            result_summary = '. '.join(list(map(lambda x: x.strip().capitalize(), result_summary.split('.'))))
            st.write(f"Here's your Summary : {result_summary}")


    elif choice == "Named Entity Recognition":
        nlp = spacy.load("en_core_web_sm")
        st.subheader("Text Based Named Entity Recognition")
        st.write(" Enter the Text below To extract Named Entities !")
        st.write("""1. Preprocessing:
Break text into words or tokens.
Tag parts of speech and linguistic features.

2. NER Model Input:
Convert text into a format suitable for NER models (e.g., word embeddings).

3. Apply NER Model:
Use a pre-trained NER model to identify entities (e.g., persons, locations, organizations).

4. Entity Classification:
Categorize identified entities into predefined classes (e.g., person names, locations, dates).""")

        raw_text = st.text_area("Your Text", "Enter Text Here")
        if raw_text != "Enter Text Here":
            doc = nlp(raw_text)
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="List of Entities")
            spacy_streamlit.visualize_parser(doc, title="Dependency Parsing")


    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment_analysis = pipeline("sentiment-analysis")
        st.write(" Enter the Text below To find out its Sentiment !")
        st.write("""1. Preprocessing:
Clean and format the text (e.g., remove special characters, normalize text).

2. Feature Extraction:
Convert text into numerical format (e.g., word embeddings, TF-IDF).

3. Train or Apply Pre-trained Model:
Train a sentiment analysis model (if using a custom model).
Or, use a pre-trained sentiment analysis model (e.g., BERT, VADER).

4. Sentiment Prediction:
Apply the model to predict sentiment for each piece of text.""")

        raw_text = st.text_area("Your Text", "Enter Text Here")
        if raw_text != "Enter Text Here":
            result = sentiment_analysis(raw_text)[0]
            sentiment = result['label']
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            if sentiment == "POSITIVE":
                st.write("""# This text has a Positive Sentiment.  """)
            elif sentiment == "NEGATIVE":
                st.write("""# This text has a Negative Sentiment. """)
            elif sentiment == "NEUTRAL":
                st.write("""# This text seems Neutral ... """)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(raw_text)

        # Display the word cloud
        st.image(wordcloud.to_array(), caption='Word Cloud', use_column_width=True)


    elif choice == "Question Answering":
        st.subheader("Question Answering")
        st.write(" Enter the Context and ask the Question to find out the Answer !")
        st.write("""
        1.Input Context and Question: Provide a piece of text (context) and a question related to that text.
        
        2.Context Understanding: The system processes and understands the context to extract relevant information.

        3.Model Inference: A pretrained QA model processes the input and generates a response.

        4.Answer Extraction: The model identifies the part of the context that contains the answer to the question.

        5.Answer Formulation: The extracted information is structured and presented as the final answer.""")
        question_answering = pipeline("question-answering")

        context = st.text_area("Context", "Enter the Context Here")

        question = st.text_area("Your Question", "Enter your Question Here")

        if context != "Enter Text Here" and question != "Enter your Question Here":
            result = question_answering(question=question, context=context)
            s1 = json.dumps(result)
            d2 = json.loads(s1)
            generated_text = d2['answer']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f" Here's your Answer :\n {generated_text}")


if __name__ == '__main__':
    main()
