# DocQuest

**DocQuest** is an AI-powered web application that automatically generates quiz questions from any PDF document. Built for educators, students, and professionals, DocQuest transforms study material, notes, or training documents into smart, interactive quizzes in seconds.

---

## 🚀 Features
- **PDF Upload:** Upload any PDF document to generate questions.
- **AI-Powered Question Generation:** Uses OpenAI GPT to create multiple-choice and true/false questions with explanations.
- **Interactive Quiz Interface:** Clean, responsive UI for answering and reviewing questions.
- **Live Statistics:** Real-time stats for questions generated, PDFs processed, and accuracy rate.
- **Modern Design:** Professional, mobile-friendly interface with smooth animations.

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **AI/LLM:** OpenAI GPT (via API)
- **Frontend:** HTML, CSS, JavaScript
- **PDF Processing:** Python PDF libraries
- **Templating:** Jinja2
- **Data Storage:** JSON (for stats)

---

## 🤖 How It Works
1. **Upload PDF:** Drag and drop or select a PDF file.
2. **Select Options:** Choose question type (MCQ/True-False), difficulty, and number of questions.
3. **Generate:** The app extracts text, sends it to OpenAI GPT, and receives questions with explanations.
4. **Quiz:** View, answer, and check explanations for each question.
5. **Stats:** See live updates on usage and performance.

---

## 🏷️ Use Cases
- Teachers and trainers creating assessments
- Students self-testing from notes or textbooks
- Professionals building training materials

---

## 📦 Setup & Run Locally
1. **Clone the repo:**
   ```bash
   git clone https://github.com/Akshatageek/DocQuest.git
   cd DocQuest
   ```
2. **Create a virtual environment & activate:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r app/requirements.txt
   ```
4. **Set your OpenAI API key:**
   - Add your OpenAI API key to your environment variables or directly in the code (not recommended for production).
5. **Run the app:**
   ```bash
   cd app
   python main.py
   ```
6. **Open in browser:**
   - Go to `http://127.0.0.1:5000`

---

## 📄 License
MIT License

---

## 🙏 Credits
- Built as part of the Skillcred GenAI session
- Powered by OpenAI GPT

---

## 🌟 Demo Screenshot
![DocQuest Screenshot](demo_screenshot.png)

---

## 💬 Feedback
Feel free to open issues or submit pull requests to improve DocQuest!
