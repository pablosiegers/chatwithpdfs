import streamlit as st

st.set_page_config(page_title="Home Page", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 300px;
        margin-left: -300px;
    }
     
    """,
    unsafe_allow_html=True,
)

st.title("Try Out - Home page")

st.sidebar.success("Choose an AI Chat.")

col_assistant, col_multiple_files = st.columns([1, 1])
with col_assistant:
    col_assistant.subheader("LangChain Chat Assistant 🦜")
    st.markdown(
        """
        **Your Personal Task Master**

        Stay organized and on top of your life with this helpful AI assistant. 
        It remembers your details and helps you:

        * **Manage Tasks:** Create, track, and prioritize your to-do list effortlessly.
        * **Set Reminders:** Never forget important appointments or deadlines again.
        * **Organize Information:** Store notes, ideas, and important details in one place.
        * **Personalized Assistance:** Get tailored suggestions and recommendations based on your preferences and habits.

        Start chatting and take control of your day!
        """
    )

with col_multiple_files:
    col_multiple_files.subheader("Chat with multiple files 💬")
    st.markdown(
        """
        **Chat with your Documents**

        Upload multiple files (PDFs, Word docs, CSVs, etc.) and have intelligent conversations with their content.
        Ask questions, summarize key points, or extract specific information. 

        Perfect for:

        * **Research:** Quickly gather insights from multiple sources.
        * **Document Analysis:**  Understand complex documents without reading them in full.
        * **Information Retrieval:** Find specific details or answers within your files.

        Upload your files and start chatting today!
        """
    )

