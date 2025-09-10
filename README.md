# 🩺 MediMind – AI Medical Chatbot

An interactive AI tool that lets you **ask medical questions** and get context-aware answers.  
Built with **LangChain, Hugging Face Transformers, ChromaDB, and Gradio**.  

---

## 🚀 Features
- Uses **LLMs + embeddings** to answer medical queries  
- **Retrieval-based QA** powered by Chroma vector DB  
- Simple **chatbot-like UI** with Gradio  
- Extensible – can be scaled with larger datasets or APIs  

---

## 📦 Installation

Clone the repo:
```bash
git clone https://github.com/username/Medi-Mind.git
cd Medi-Mind
```

Create a virtual environment (recommended):
```bash
python -m venv venv
```

On Linux/Mac:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the app:
```bash
python main.py
```

This will start a **Gradio interface** in your browser where you can chat with MediMind.  

---

## 🛠️ Tech Stack
- **LangChain** – for chaining LLM and retrieval  
- **Gradio** – chatbot UI  
- **Hugging Face Models** – for text generation  
- **ChromaDB** – vector database for context retrieval  

---

## 📌 Example Questions
1. What is anemia?  
2. What are the symptoms of diabetes?  
3. Explain hypertension in simple terms.  
4. How is asthma treated?  

---

## 📄 License
This project is licensed under the MIT License.  

---

💡 *Contributions are welcome! Fork the repo and improve MediMind.*  
