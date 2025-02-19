import streamlit as st
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import spacy
from transformers import pipeline 

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load SpaCy model for NLP (though it's not being used here, but you can extend if needed)
nlp = spacy.load('en_core_web_sm')

# Function to fetch the first Wikipedia or SATP link
def get_first_google_link(gname):
    for result in search(gname, num_results=10):
        if 'wikipedia.org' in result:
            return result
        elif 'satp.org' in result:
            return result
    return None

# Function to extract the textual content from the link
def get_textual_content(group_name):
    first_link = get_first_google_link(group_name)
    if first_link:
        url = first_link
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p') 
            
            # Extract text from the paragraphs
            text = ' '.join([para.get_text() for para in paragraphs])
            
            # Split the text into sentences
            sentences = text.split('. ')
            
            # Return the first four sentences
            op = '. '.join(sentences[:4]) + '.' if len(sentences) > 1 else None
            return op
        else:
            return None
    else:
        return None

# Function to classify the group based on the extracted content
def classify_group(text):
    classes_labels = ["Religious Terrorist Group", "Separatist Terrorist", "Left-Wing Terrorist Group", "Right-Wing Terrorist Group", "Non terrorist Group"]
    result = classifier(text, classes_labels)
    return result['labels'][0]

# Streamlit App
def main():
    st.title("Terrorist Group Classification App")
    
    # User input for group name
    group_name = st.text_input("Enter the terrorist group name", "")
    
    if group_name:
        # Fetch textual content based on the group name
        content = get_textual_content(group_name)
        
        if content:
            st.write(f"### Textual content fetched for group '{group_name}':")
            st.write(content)
            
            # Classify the group
            classification = classify_group(content)
            st.write(f"### The group '{group_name}' is classified as: {classification}")
        else:
            st.error("Couldn't fetch data for the given group name. Please try again with a different name.")
    
    # Add some information or instructions
    st.sidebar.header("Instructions")
    st.sidebar.text("""
    1. Enter the name of a terrorist group in the input box above.
    2. The app will fetch information from Wikipedia (or other related sources) and classify the group.
    3. The classification will indicate the type of terrorist group based on the content.
    """)

if __name__ == "__main__":
    main()
