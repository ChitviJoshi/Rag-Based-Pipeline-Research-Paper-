from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from transformers import pipeline

# === Step 1: Load PDF ===
pdf = PdfReader("sample.pdf")
text = ""
for i, page in enumerate(pdf.pages):
    content = page.extract_text()
    if content:
        text += content
    print(f"✅ Page {i+1}: {len(content) if content else 0} characters")

print("\n📄 First 500 characters of text:")
print(text[:500])
print(f"🔢 Total characters extracted: {len(text)}")

# === Step 2: Split into Chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)

print(f"\n✅ Total chunks created: {len(chunks)}")
print("🔹 Preview of first chunk:\n", chunks[0][:300])

# === Step 3: Generate Embeddings ===
print("\n🔄 Generating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

# === Step 4: Setup GPT-2 Pipeline (safe)
print("\n🚀 Loading Loading FLAN-T5 (google/flan-t5-base)...")
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150
)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

# === Step 5: Build Retrieval Chain with safe input length ===
prompt_template = PromptTemplate.from_template(
    "Based on the following text, provide a detailed and specific answer to the question.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


llm_chain = LLMChain(llm=llm, prompt=prompt_template)
combine_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # only 1 chunk to avoid token overflow
qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_chain)

# === Step 6: Ask a Question ===
question = "What is the topic of this document?"
print(f"\n❓ Question: {question}")
answer = qa_chain.invoke(question)

print("\n🤖 Answer:\n", answer)
