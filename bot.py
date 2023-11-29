import json
import os

from flask import Flask, render_template
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS

from common import prompt_template, model_path, model_name
from vectorizer import createVectorDB


def setCustomPrompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt


def loadLLM():
    print("loading LLM...")
    llm = CTransformers(
        model=model_path + model_name,
        model_type="llama",
        config={'max_new_tokens': 1024,
                'temperature': 0.01,
                'context_length': 2048}
    )
    return llm


def retrivalChain(llm, prompt, db):
    conv_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return conv_chain


def qa_bot():
    print("Embedding...")
    createVectorDB()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local("../../smartims_data/", embeddings)
    llm = loadLLM()
    prompt = setCustomPrompt()

    qa = retrivalChain(llm, prompt, db)
    return qa


def logic(bot, msg):
    print('Querying...')
    qa = bot
    response = qa({'query': msg})
    print('Got a response!')
    return response


llama_bot = qa_bot()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask/<question>')
def ask(question):
    result = {}
    response = logic(llama_bot, question)
    result['question'] = question
    result['answer'] = response['result']
    return json.dumps(result)


print("server started")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5005)))
