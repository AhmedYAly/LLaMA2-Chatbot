from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from common import db_path


def createVectorDB():
    loader = CSVLoader("../../smartims_data/pages_contents.csv", csv_args={"delimiter": ","}, encoding="utf8")
    raw_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    data = text_splitter.split_documents(raw_data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={"normalize_embeddings": True})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(db_path)
