from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Helper function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Vector store setup
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available, respond with 'answer is not available in the context.'\n
    Context:\n{context}\n
    Question: {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# YouTube transcript extraction and summarization
def extract_transcript_details(youtube_video_url):
    video_id = youtube_video_url.split("=")[1]
    transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join([item["text"] for item in transcript_text])
    return transcript

def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    if request.method == 'POST':
        user_question = request.form['user_question']
        pdf_files = request.files.getlist("pdf_files")

        # Save uploaded PDFs and extract text
        pdf_paths = []
        for pdf in pdf_files:
            filename = secure_filename(pdf.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf.save(path)
            pdf_paths.append(path)

        raw_text = get_pdf_text(pdf_paths)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]

        return render_template('pdf_chat.html', answer=answer)

    return render_template('pdf_chat.html', answer=None)

@app.route('/yt_summary', methods=['GET', 'POST'])
def yt_summary():
    if request.method == 'POST':
        youtube_link = request.form['youtube_link']
        transcript_text = extract_transcript_details(youtube_link)
        prompt = "You are a YouTube video summarizer. Summarize this video in 250 words:\n"
        summary = generate_gemini_content(transcript_text, prompt)
        return render_template('yt_summary.html', summary=summary, youtube_link=youtube_link)

    return render_template('yt_summary.html', summary=None, youtube_link=None)

if __name__ == '__main__':
    app.run(debug=True)
