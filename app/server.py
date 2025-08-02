import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP

load_dotenv()
mcp = FastMCP(name="RAG_mcp_agent")


def load_documents():
    # Load documents from the specified directory
    current_dir = os.path.dirname(__file__)
    docs_dir = os.path.join(current_dir, "../documents")
    if not docs_dir:
        raise ValueError("Documents directory not found. Please check the path.")
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
    )
    documents = loader.load()
    return documents


def create_vector_store():
    docs = load_documents()
    text_splitter = CharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
    )
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


@mcp.tool()
def read_document(query: str):
    """To read a personal document and return the answer based on the query. query harus berupa pertanyaan."""

    system_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Kamu adalah asisten pribadi yang dirancang untuk membantu menjawab pertanyaan berdasarkan dokumen pribadi saya.
    Tugas utama Anda adalah memberikan informasi yang akurat dan relevan berdasarkan konteks dokumenyang telah disediakan. 

    Berikut adalah isi dokumen:
    {context}
    -------
    Berikut pertanyaan yang harus Anda jawab:
    {question}

    Jika pertanyaan tidak dapat dijawab dengan informasi yang ada, berikan respons yang sesuai.
""",
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    vectorstore = create_vector_store()
    retriever = vectorstore.as_retriever()
    agent = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": system_prompt},
    )

    return agent.run(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
