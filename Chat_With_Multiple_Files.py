import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
from PyPDF2 import PdfReader
import docx
import csv
import io

def create_vectorstore(uploaded_docs):
    # Get PDF text
    raw_text = get_raw_text(uploaded_docs)
    #st.write(raw_text) # To print text in pdfs

    # Get Text Chunks
    text_chunks = get_text_chunks(raw_text)
    #st.write(text_chunks) # To print the chunks

    # Create Vector Store
    embeddings = OpenAIEmbeddings() # Paid method
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") # free
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_raw_text(uploaded_docs):
    try:
        text = ""
        for doc in uploaded_docs:
            # Get file name and extension
            file_path = Path(doc.name)

            # If PDF then uses Pdf Reader
            if(file_path.suffix == ".pdf"):
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
            
            # if .txt then uses .read function
            elif(file_path.suffix == ".txt"):
                text += doc.read().decode() + '\n'

            # if word file then uses python-docx library
            elif(file_path.suffix == ".docx"):
                word_doc = docx.Document(doc)
                for para in word_doc.paragraphs:
                    text += para.text + '\n'
            
            # if excel file then uses csv and io libraries
            elif(file_path.suffix == ".csv"):
                excel_doc = doc.read().decode("utf-8")
                csv_reader = csv.reader(io.StringIO(excel_doc))
                for row in csv_reader:
                    text += ','.join(row) + '\n'
            
            text += "\n\n"
            text += "###"
            text += "\n\n"

    except Exception as e:
        st.error(f"Error reading the file: {e}")

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["###"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():

    # Retrieving OpenAI_API_KEY
    load_dotenv()
    
    # Setting application title
    st.set_page_config(page_title="MRAG App", page_icon=":books:", layout="wide")
    st.title("💬 Chat with multiple files")

    # Initialize variables in session_state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # We add the file uploader component from streamlit to accept multiple files
    uploaded_docs = st.file_uploader("Upload your files: .pdf, .txt, .docx, .csv", accept_multiple_files=True)    

    # If there are no messages in the chat then we add the assistant message
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello, how can I help you?"}]

    # We show the messages in the chat message component from streamlit
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
         
    # Once the user sends a prompt we execute the following, is disabled if no documents have been uploaded
    if prompt := st.chat_input(placeholder="What skills does Bob have?", disabled= not uploaded_docs):

        # We add the user prompt to the messages and chat_history components and write it in chat_message component
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Processing"):

            # Create vectorstore
            vectorstore = create_vectorstore(uploaded_docs)

            # Cretate conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
        
            # Send prompt and get response from llm
            response = st.session_state.conversation({
                'question': 
                    """
                    Answer the question based only on the following context:
                    You are a human resource assistant. Your job is to respond to the user's prompt:
                    """ + prompt
            })
            
            # st.write(response) # debug command to print question or prompt, chat history and answer
            msg = response['answer']

            # Add response to messages and chat_history components and write it in the chat_message component
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)


# Initializing application by executing main function
if __name__ == '__main__':
    main()