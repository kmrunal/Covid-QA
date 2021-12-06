import pandas as pd
import streamlit as st
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline


@st.cache(allow_output_mutation=True, show_spinner=False)
def read_data(data=r"C:\Users\admin\Desktop\VT\ISR\project1\covid_qa.csv"):
    """Read the data from local."""
    print("read")
    df = pd.read_csv(r"C:\Users\admin\Desktop\VT\ISR\project1\covid_qa.csv")
    df = df.rename(columns={'Context': 'text', 'Answers': 'content'})
    document_store = InMemoryDocumentStore()
    document_store.write_documents(df.to_dict(orient="records"))
    retriever = TfidfRetriever(document_store=document_store)
    return retriever


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    print("in model")
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2-covid", use_gpu=False)
    return reader

@st.cache(allow_output_mutation=True, show_spinner=False)
def getPipeline():
    print("in pipeline")
    pipe = ExtractiveQAPipeline(load_bert_model(), read_data())
    return pipe


def main():
    data = read_data()
    model = load_bert_model()
    pipe = getPipeline()

    st.markdown(
        "<h1 style='text-align: center; '>Covid-19 search engine</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<h2 style='text-align: center; '>Stay safe</h2>", unsafe_allow_html=True)

    search_query = st.text_input(
        "Search for Covid-19 here", value="", max_chars=None, key=None, type="default"
    )

    if search_query != "":
        prediction = pipe.run(
            query=search_query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )

        res = prediction
        res = res['answers']
        if res:
            for r in res:
                if r.answer:
                    st.markdown(
                            f"<div style='text-align: left; background:#f8f1eb; margin-top: 1rem; margin-bottom: 1rem; padding: 1rem;'> <div style='margin-bottom: 1rem;'><h5 style='text-align: left; '>Answer</h5>{r.answer}</div></div>",
                            unsafe_allow_html=True,
                    )
                if r.context:
                    st.markdown(f"<h5 style='text-align: left; '>Context</h5>", unsafe_allow_html=True,)

                    st.markdown(
                            f"<div style='text-align: left; background:#f8f1eb; margin-top: 1rem; margin-bottom: 1rem; padding: 1rem;'> <div style='margin-bottom: 1rem;'>{r.context}</div></div>",
                            unsafe_allow_html=True,
                    )

        else:
            st.markdown(
                "<h5 style='text-align: center; '>No Search results, please try again with different keywords</h5>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
