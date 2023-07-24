from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain

st.set_page_config(page_title="CHILD Project", page_icon=":robot_face:")

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split
@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name='langchain-pdf'
    )
    return doc_db
llm = ChatOpenAI()
doc_db = embedding_db()
def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result
def main():
    st.markdown("<h1 style='text-align: center;'>CHILD Projects</h1>",
                unsafe_allow_html=True)
    st.image('./banner.jpg')
    st.text("Query on any topics related to healthcare and innovation")

    text_input = st.text_input("Ask your query about any CHILD project. Code adapted from open source, built by Andrew Soh") 
    if st.button("Ask Query"):
        if len(text_input)>0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)
            
if __name__ == "__main__":
    main()
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'cost' not in st.session_state:
        st.session_state['cost'] = []
    if 'total_tokens' not in st.session_state:
        st.session_state['total_tokens'] = []
    if 'total_cost' not in st.session_state:
        st.session_state['total_cost'] = 0.0
    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.image("./child logo_with brushstroke.png", use_column_width=True)
    with st.sidebar:
        st.caption("Developed from open source codes for a search based on CHILD collection of projects.")
        st.caption("Do be aware that there will be longer waiting time as the AI search through our knowledge base")
    def open_page(url):
        open_script= """
            <script type="text/javascript">
                window.open('%s', '_blank').focus();
            </script>
        """ % (url)
        html(open_script)

    st.button('Open link', on_click=open_page, args=('https://streamlit.io',))
        st.button('check out our website [link](https://child.chi.sg)')
        st.write("check out our website [link](https://child.chi.sg)")
