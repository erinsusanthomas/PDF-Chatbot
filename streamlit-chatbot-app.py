import streamlit as st 
import os
import shutil
import time
from RAG_code import *

__import__('pysqlite3')
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Page configurations
st.set_page_config(page_title="Chat with Your PDFs", layout="wide", initial_sidebar_state="expanded")
# Heading
st.header("Chat with Your PDFs: Upload & Get Answers")

# Function to make temporary folders
def create_temp_folder(folder):
    if os.path.exists(folder):
        # Remove all contents of the folder
        shutil.rmtree(folder)
    # Create a new folder
    os.makedirs(folder)
upload_folder = 'test_data'
store_dir = "test_db_store"
# Check if the temp folders are already created
if 'folders_initialized' not in st.session_state:
    create_temp_folder(upload_folder)
    create_temp_folder(store_dir)
    # Mark folders as initialized to prevent rerun
    st.session_state['folders_initialized'] = True
    
# Initialize session state variables to store uploaded file names
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []
# Initialize session state variables to store uploaded file count
if 'ctr' not in st.session_state:
    st.session_state['ctr'] = 0

# File uploader function
def uploader():
    # Upload file (can upload multiple files, one-by-one)
    uploaded_file = st.file_uploader('**Upload your file**', label_visibility="collapsed",
                                     type="pdf", key='file_uploader')
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        # If the file has not been uploaded previously, append to session state
        if file_name not in st.session_state['uploaded_files']:
            # new upload = stored in session - no | stored in db - yes/no
            st.session_state['uploaded_files'].append(file_name)
            st.session_state['ctr'] += 1
            print(f'File "{file_name}" uploaded successfully! Uploaded file #{st.session_state["ctr"]}')
        else:
            # duplicate upload = stored in session - yes | stored in db - yes (so, no need to check existence in db)
            duplicate_alert = st.warning(f'File "{file_name}" is a duplicate and was not uploaded. \
                       If this is a different file, please change the file name to upload it.')
            time.sleep(3) # wait for 3 seconds
            duplicate_alert.empty() # clear the alert
            return

    if not st.session_state['uploaded_files']:
        st.warning("Please upload files before you start asking questions")
    else:
        start_process_time = time.time()

        if uploaded_file is not None:
            file_path = os.path.join(upload_folder, uploaded_file.name)
            # Check if the file does not exist
            # new upload  = stored in session - no | stored in db - yes
            if os.path.exists(file_path):
                exists_alert = st.warning(f"File '{uploaded_file.name}' already exists in your database. \
                                            If this is a different file, please change the file name to upload it.")
                time.sleep(3) # wait for 3 seconds
                exists_alert.empty() # clear the alert
            # new upload  = stored in session - no | stored in db - no ==> so, we need to process to store in db
            else: 
                bytes_data = uploaded_file.read()
                # Saving the uploaded file to the data folder
                # with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
                with open(file_path, "wb") as f:
                    f.write(bytes_data)
                upload_alert = st.success(f"'{uploaded_file.name}' uploaded and saved successfully!")
                time.sleep(3) # wait for 3 seconds
                upload_alert.empty() # clear the alert
                # Processing the file
                with st.spinner('Processing files...'):
                    # Process the uploaded files
                    documents = load_documents(upload_folder)
                    chunks = split_documents(documents)
                    add_to_chroma(store_dir, chunks)
                    process_alert = st.success(f"File '{uploaded_file.name}' has been processed successfully! \
                                            You can now ask questions.")
                    time.sleep(3) # wait for 3 seconds
                    process_alert.empty() # clear the alert
        end_process_time = time.time()
        process_duration = end_process_time - start_process_time
        print(f"Time taken to process files: {process_duration:.2f} seconds")

uploader()

# Add sidebar to list uploaded files
with st.sidebar:
    st.header("Uploaded Documents")
    if st.session_state["uploaded_files"]:
        for uploaded_file in st.session_state["uploaded_files"]:
            st.write(f"- {uploaded_file}")
    else:
        st.write("No documents uploaded")

# Initialize session states to store chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
# Display the chat messages from session state
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session states to flag to disable chat input
if "input_disabled" not in st.session_state:
    st.session_state["input_disabled"] = False  
# Function to disable chat input
def disable():
    if st.chat_input:
        st.session_state["input_disabled"] = True

# Capture the chat input from the user
user_input = st.chat_input("Ask your question here...", disabled=st.session_state["input_disabled"], 
                           on_submit = disable, key="input")

# When user submits input: process the input, disable chat input, and fetch the bot response
if user_input:
    # Append the user's message to session state
    st.session_state['messages'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Bot's response
    with st.spinner('Fetching response...'):
        start_fetch_time  = time.time()
        bot_response = get_response(store_dir, user_input) 
        end_fetch_time  = time.time()
        fetch_duration  = end_fetch_time  - start_fetch_time 
        print(f"Time taken to fetch response: {fetch_duration :.2f} seconds")

    # Append the bot's message to session state
    st.session_state['messages'].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Re-enable input after fetching response
    st.session_state["input_disabled"] = False
    st.rerun()