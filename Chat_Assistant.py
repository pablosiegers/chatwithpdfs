
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from streamlit_option_menu import option_menu
from datetime import datetime
import io

# Get the project name to use in path variables
project_folder_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))  # Three levels up

# Env
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY

def main():

    # Page Config 
    st.set_page_config(page_title="LangChain Chat Assistant 🦜", layout="wide")
    st.title("Try Out - LangChain Chat Assistant 🦜")
    
    # LLM
    LLM = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini", 
        openai_api_key=OPENAI_API_KEY, 
        streaming=True
    )

    with st.sidebar:
        # Language Selection Dropdown
        if 'language' not in st.session_state:
            st.session_state['language'] = "English"  # Default language

        selected_language = st.selectbox("Select Language", ["English", "Spanish", "Dutch", "Japanese", "French", "Italian"])
        if selected_language:
            st.session_state['language'] = selected_language

    history_chat_context = []
    tab_selected = ""

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "user_memory" not in st.session_state:
        st.session_state["user_memory"] = ""

    if "audio_bytes" not in st.session_state:
        st.session_state["audio_bytes"] = ""

    selected = option_menu(
        menu_title = None,
        options= ["Chat", "Voice"],
        icons=["cast", "mic"],
        default_index=0,
        orientation= "horizontal"
    )

    if selected == "Chat":
        tab_selected = "messages1"
        with open(os.path.join("..", project_folder_name, "data", "user-memory", "user-memory.txt"), 'r') as doc:
                user_memory = doc.read()
        # If there are no messages in the chat then we add the assistant message
        if tab_selected not in st.session_state:
            st.session_state[tab_selected] = [{"role": "assistant", "content": start_message_chat(LLM, user_memory, selected_language)}]
            history_chat_context += st.session_state[tab_selected]
        # We show the messages in the chat message component from streamlit
        for msg in st.session_state.messages1:
            st.chat_message(msg["role"]).write(msg["content"])

    if selected == "Voice":
        tab_selected = "messages2"
        with open(os.path.join("..", project_folder_name, "data", "user-memory", "user-memory.txt"), 'r') as doc:
                user_memory = doc.read()
        # If there are no messages in the chat then we add the assistant message
        if tab_selected not in st.session_state:
            st.session_state[tab_selected] = [{"role": "assistant", "content": start_message_chat(LLM, user_memory, selected_language)}]
            history_chat_context += st.session_state[tab_selected]

        '''Click the microphone icon to start recording; click again to stop.'''
        audio_bytes = audio_recorder(
            pause_threshold=3.0,
            text="Record button ->",
            recording_color="#ff0000",
            icon_size="x2",
        )
        
        if audio_bytes:
            #st.audio(audio_bytes, format="audio/wav")
            save_audio_file(audio_bytes)           

            # Find the newest audio file
            audio_file_path = os.path.join("..", project_folder_name, "data", "last-audio", "user_audio.mp3")
            audio_transcription = transcribe_audio(audio_file_path)
            user_voice = audio_transcription.text

            messageforchat = f"Processing voice prompt ... 💫"
            voice = True
            get_response(LLM,user_voice, user_memory, history_chat_context, voice, messageforchat, tab_selected, selected_language)
        
        # We show the messages in the chat message component from streamlit
        for msg in st.session_state.messages2:
            st.chat_message(msg["role"]).write(msg["content"])
        

    # Initialize variables in session_state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if user_prompt := st.chat_input(placeholder="Ask anything about reminders, pending tasks, etc"):
        messageforchat = f"Processing user prompt ... 💫"
        voice = False
        get_response(LLM, user_prompt, user_memory, history_chat_context, voice, messageforchat, tab_selected, selected_language)

def get_actual_date_and_time():
    now = datetime.now()
    return now.strftime("%d-%m-%Y %H:%M:%S")

def update_user_memory(LLM, user_memory, history_chat_context):

    current_datetime = get_actual_date_and_time()

    # Template and Chain
    UPDATEMEMORY = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """  
                Based on the Actual User Memory provided below, please respond exclusively with the updated user memory. 
                Ensure that your response adheres to the following format:
                -------------------------------------------
                Format: 
                - Last date modified: {current_datetime}
                - User details:
                    (List user details like name, age, etc, update or remove if necessary)
                - Hobbies:
                    (list Hobbies, update or remove if necessary)
                - Reminders:
                    (list reminders, date and time, update or remove if necessary)
                - Pending tasks:
                    (list pending tasks, update or remove if necessary)
                - Gym schedule:
                (...)
                -------------------------------------------
                If you determine that there is no need to update the user memory, please replicate the actual user memory in your response.
                Please find the following important parameters separated:
                -------------------------------------------
                Actual User Memory: <-{user_memory}->"
                -------------------------------------------
                Chat History: <-{history_chat_context}->"
                -------------------------------------------
                """,
            ),
            MessagesPlaceholder(variable_name="history_chat_context"),
        ]
    )

    interview_chain = (
        {"user_memory": itemgetter("user_memory"),
         "current_datetime": itemgetter("current_datetime"),
         "history_chat_context": itemgetter("history_chat_context")}
        | UPDATEMEMORY 
        | LLM 
        | StrOutputParser()
    )   

    return interview_chain.invoke({"user_memory": user_memory, "current_datetime": current_datetime, "history_chat_context": history_chat_context})


def get_response(LLM, user_prompt, user_memory, history_chat_context, voice, messageforchat, tab_selected, selected_language):
    # We add the user prompt to the messages and chat_history components and write it in chat_message component
    if(tab_selected == "messages1"):
        st.session_state.messages1.append({"role": "user", "content": user_prompt})
    elif(tab_selected == "messages2"):
        st.session_state.messages2.append({"role": "user", "content": user_prompt})
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    if (voice == False):
        st.chat_message("user").write(user_prompt)

    history_chat_context += st.session_state[tab_selected]

    try:
        with st.spinner(messageforchat):
            response = user_memory_chain(LLM, user_prompt, user_memory, history_chat_context, selected_language)
            # Add response to messages and chat_history components and write it in the chat_message component
            if(tab_selected == "messages1"):
                st.session_state.messages1.append({"role": "assistant", "content": response})
            elif(tab_selected == "messages2"):
                st.session_state.messages2.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            if (voice == False):
                st.chat_message("assistant").write(response)
            history_chat_context += st.session_state[tab_selected]
            updated_user_memory = update_user_memory(LLM, user_memory, history_chat_context)
            save_user_memory_file(updated_user_memory)
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Text-to-speech part
    if voice == True:  # Only for chat input and in the chat tab
        try:
            # Create an API client
            client = OpenAI(api_key=OPENAI_API_KEY)

            # Generate audio from the assistant's response
            response_audio = client.audio.speech.create(
                model="tts-1",
                voice="alloy", 
                input=response
            )

            # Store the audio bytes in memory
            audio_bytes = io.BytesIO(response_audio.content)
            audio_bytes.seek(0)

            # Play the audio in Streamlit
            st.audio(audio_bytes, format="audio/mp3") 

        except Exception as e:
            st.error(f"An error occurred during text-to-speech: {e}")

def transcribe_audio(audio_file):
    # Create an api client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load audio file
    audio_file= open(os.path.join(os.getcwd(), audio_file), "rb")

    # Transcribe
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )

    return transcription

def save_user_memory_file(updated_user_memory):
    directory = os.path.join("..", project_folder_name, "data", "user-memory")
    file_name = f"user-memory.txt"
    full_path = os.path.join(directory, file_name)

    with open(full_path, "wb") as f:
        f.write(updated_user_memory.encode('utf-8'))

    return file_name

def save_audio_file(audio_bytes):
    directory = os.path.join("..", project_folder_name, "data", "last-audio")
    file_name = f"user_audio.mp3"
    full_path = os.path.join(directory, file_name)

    with open(full_path, "wb") as f:
        f.write(audio_bytes)

    return file_name

def user_memory_chain(LLM, user_prompt, user_memory, history_chat_context, selected_language):

    # Template and Chain
    ANSWERPROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """  
                You are a highly intelligent and capable personal assistant. 
                Your primary goal is to assist the user in managing their tasks, staying organized, and informed.              
                If user request a reminder please make sure it has date and time to be reminded.
                If date is expired, do not accept it.

                Please find the following parameters separated:
                -------------------------------------------
                User Prompt: <-{user_prompt}->"
                -------------------------------------------
                User Memory: <-{user_memory}->"
                -------------------------------------------
                Chat History: <-{history_chat_context}->"
                -------------------------------------------
                Your response has to be in the following language: <-{selected_language}->"
                -------------------------------------------
                """,
            ),
            MessagesPlaceholder(variable_name="history_chat_context"),
        ]
    )

    interview_chain = (
        {"user_memory": itemgetter("user_memory"),
         "user_prompt": itemgetter("user_prompt"),
         "selected_language": itemgetter("selected_language"),
         "history_chat_context": itemgetter("history_chat_context")}
        | ANSWERPROMPT 
        | LLM 
        | StrOutputParser()
    )   

    return interview_chain.invoke({"user_prompt": user_prompt, "user_memory": user_memory, "selected_language": selected_language, "history_chat_context": history_chat_context})

def start_message_chat(LLM, user_memory, selected_language):

    # Template and Chain
    FIRSTANSWER = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a highly intelligent and capable personal assistant. 
                Your primary goal is to assist the user in managing their tasks, staying organized, and informed. 
                
                --------------------------------------
                User Memory: <-{user_memory}->"
                --------------------------------------
                Your response has to be in the following language: <-{selected_language}->"
                -------------------------------------------
                """,
            )
        ]
    )

    interview_chain = (
        {"user_memory": itemgetter("user_memory"),
        "selected_language": itemgetter("selected_language")}
        | FIRSTANSWER 
        | LLM 
        | StrOutputParser()
    )   

    return interview_chain.invoke({"user_memory": user_memory, "selected_language": selected_language})


if __name__ == '__main__':
    main()