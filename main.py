
# Import the necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
import openai
import re

# Initialize your OpenAI API credentials
openai.api_key = st.secrets["openai_api_key"]

# This is a helper function to read PDFs
def read_pdf(pdf, pages):
    text = ""
    for page in pages:
        text += pdf.pages[page].extract_text()
    return text

def ask_gpt3(question, context, temperature, max_tokens, top_p, frequency_penalty, role):
    message = [
        {"role": "system", "content": "You have the following information from the paper: "+context},
        {"role": role, "content": question}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty
    )
    return response['choices'][0]['message']['content']

def main():
    st.title('🧠 GPT-3 for Scientific Papers')
    st.markdown('An application that allows you to upload a scientific paper, then the AI will read it and you can ask questions about the content of the paper.')

    # Configure the file uploader
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        # Read the text of the PDF
        pdf = PdfReader(uploaded_file)
        num_pages = len(pdf.pages)

        # Configure the parameters of the OpenAI API
        st.sidebar.title('🛠️ OpenAI API Configuration')
        page_selection = st.sidebar.multiselect('Pages', options=range(1, num_pages+1), default=range(1, num_pages+1))        
        st.sidebar.markdown("<small>Select the pages you want to use as context</small>", unsafe_allow_html=True)
        temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5)
        st.sidebar.markdown("""<small>Temperature determines the randomness of the AI's responses. A higher value will make the responses more diverse, but also riskier.</small>""", unsafe_allow_html=True)
        max_tokens = st.sidebar.slider('Max Tokens', min_value=10, max_value=500, value=150)
        st.sidebar.markdown("""<small>Max tokens limit the length of the AI's response.</small>""", unsafe_allow_html=True)
        top_p = st.sidebar.slider('Top P', min_value=0.0, max_value=1.0, value=0.9)
        st.sidebar.markdown("""<small>Top P is the cumulative probability by the highest-ranking words, which affects the diversity of the response.</small>""", unsafe_allow_html=True)
        frequency_penalty = st.sidebar.slider('Frequency Penalty', min_value=-2.0, max_value=2.0, value=0.0)
        st.sidebar.markdown("""<small>Frequency penalty reduces the likelihood of frequent words.</small>""", unsafe_allow_html=True)
        role = st.sidebar.selectbox('Role', ('system', 'user', 'assistant'), index=2)
        st.sidebar.markdown("""<small>The role defines the behavior of the chatbot.</small>""", unsafe_allow_html=True)

        

        # Convert the page selection to 0-indexed for use with PyPDF2
        page_selection = [page-1 for page in page_selection]

        # Use the helper function to extract the text
        context = read_pdf(pdf, page_selection)

        # Create a text input field for the question
        question = st.text_input("Enter your question here")

        if st.button('Ask the question'):
            if question:
                try:
                    # Use the OpenAI API to get an answer
                    response = ask_gpt3(question, context, temperature, max_tokens, top_p, frequency_penalty, role)

                    # Display the answer
                    st.markdown("**Answer:**")
                    st.markdown(response)
                except openai.error.InvalidRequestError as e:
                    # Extract the number of requested tokens and the maximum allowed from the error message
                    max_tokens, tokens_requested = re.findall(r'\d+', str(e))
                    st.error(f"You have requested {tokens_requested} tokens when the maximum allowed is {max_tokens}. Please reduce the number of pages in the configuration bar.")
            else:
                st.warning('Please, enter a question.')

if __name__ == '__main__':
    main()