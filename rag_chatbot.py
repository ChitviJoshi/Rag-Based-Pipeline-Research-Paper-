import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from transformers import pipeline

st.set_page_config(page_title="Research Paper Q&A", layout="wide")
st.title("📄 RAG-Based Research Paper Assistant")

uploaded_file = st.file_uploader("Upload a research paper PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📚 Reading and processing PDF..."):
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content

    st.success(f"✅ Extracted {len(text)} characters from PDF.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    st.info(f"🔹 Created {len(chunks)} text chunks for retrieval.")

    # Embeddings & vector store
    with st.spinner("🔍 Generating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Load FLAN-T5 model
    with st.spinner("🤖 Loading FLAN-T5 model..."):
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=150,
            device=-1  # CPU; change if GPU is available
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # === Dynamic prompt template selector ===
    def get_prompt_template(question: str) -> PromptTemplate:
        q = question.lower()
        if "author" in q or "written by" in q:
            return PromptTemplate.from_template(
                "Based on the academic text, list only the authors of the current research paper. Do not include any cited references. "
                "Focus on names and emails usually found at the top of the document, before the abstract.\n\n"
                "=== Context ===\n{context}\n\n=== Question ===\n{question}\n\n=== Authors of this paper ==="
            )

        elif "summary" in q or "summarize" in q:
            return PromptTemplate.from_template(
                "You are a research summarizer. Write a clear, 2-3 sentence summary of the main objective and findings of this research paper.\n\n"
                "=== Context ===\n{context}\n\n=== Question ===\n{question}\n\n=== Summary ==="
            )
        elif "citation" in q or "reference" in q:
            return PromptTemplate.from_template(
                "List the major references or citations in this research paper. Focus only on important prior works the paper builds on.\n\n"
                "=== Context ===\n{context}\n\n=== Question ===\n{question}\n\n=== Citations ==="
            )
        else:
            return PromptTemplate.from_template(
                "You are a research assistant. Answer the question based on the academic context below.\n\n"
                "=== Context ===\n{context}\n\n=== Question ===\n{question}\n\n=== Answer ==="
            )

    # Q&A Interface
    question = st.text_input("Ask a question about the paper:")
    if question:
        with st.spinner("💬 Thinking..."):
            prompt_template = get_prompt_template(question)
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)
            combine_chain = StuffDocumentsChain(
                llm_chain=llm_chain, document_variable_name="context"
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
            qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_chain)

            answer = qa_chain.run(question)
            st.markdown("### ✅ Answer:")
            st.write(answer)
