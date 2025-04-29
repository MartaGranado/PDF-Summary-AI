import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import textwrap
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_together import ChatTogether
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")  # Store your Together key in .env
os.environ["TOGETHER_API_KEY"] = api_key

# Initialize Flask app
app = Flask(__name__)

# Set file upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to summarize PDF
def summarize_pdf(pdf_path):
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Load LLM using Together.ai (e.g., Mixtral or LLaMA2)
        llm = ChatTogether(
            together_api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.3
        )

        # Summarize
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.invoke(docs)

        # Extract summary text
        summary_text = summary.get("output_text", "")
        if not summary_text:
            return None

        # Wrap the text to a specified word limit (e.g., 120 words per line)
        word_limit = 120
        wrapped_summary = textwrap.fill(summary_text, width=word_limit)
        
        return wrapped_summary
    
    except Exception as e:
        print(f"Error summarizing PDF: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request")
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return 'No selected file'
    
    print(f"Received file: {file.filename}")  # Debugging line
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        summary_name = request.form.get('summary_name', 'summary')  # Default summary name if not provided
        summary_file = f"{summary_name}.txt"
        summary_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_file)
        
        # Summarize the PDF
        summary_text = summarize_pdf(file_path)
        if summary_text:
            # Write summary to the file
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            return send_from_directory(app.config['UPLOAD_FOLDER'], summary_file, as_attachment=True)
        else:
            return 'Error generating summary'
    return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=True)
