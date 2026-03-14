# DocQuest

AI-powered quiz generator that turns PDF content into multiple-choice and true/false questions using OpenAI.

## Why This Project Is Useful
DocQuest helps educators and students quickly create practice quizzes from notes, handouts, and lecture PDFs.

## Features
- PDF upload and text extraction
- AI-generated MCQ + True/False questions
- Difficulty selection: Easy, Medium, Hard
- Optional answer explanations
- Results view with answer checking
- Quiz PDF export
- Usage statistics endpoint
- Fallback generator when API is unavailable

## Tech Stack
- Python 3.10+
- Flask
- Gemini API or OpenAI API (model configured via env)
- PyPDF2
- ReportLab
- HTML, CSS, JavaScript

## Project Structure
```text
DocQuest/
   app/
      main.py
      templates/
         index.html
         results.html
      static/
         styles.css
      uploads/
      stats.json
   requirements.txt
   Procfile
   render.yaml
   .env.example
   README.md
```

## Local Setup
1. Clone repo
```bash
git clone https://github.com/Akshatageek/DocQuest.git
cd DocQuest
```

2. Create virtual env
```bash
python -m venv .venv
```

3. Activate env
Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Configure environment variable
- Preferred: create .env at repository root
```env
OPENAI_API_KEY=your_api_key_here
```

- Or for current Windows PowerShell session:
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

6. Run app
```bash
cd app
python main.py
```

Open: http://127.0.0.1:5000

## Environment Variables
- AI_PROVIDER: `gemini`, `openai`, or `auto`
- GEMINI_API_KEY: required when using Gemini
- GEMINI_MODEL: optional (default `gemini-1.5-flash-latest`)
- OPENAI_API_KEY: required for OpenAI-based generation
- OPENAI_MODEL: optional (default `gpt-4o-mini`)

## API Endpoints
- GET /health
- GET /api/stats
- POST /generate
- POST /sample
- POST /download/pdf

## Deploy on Render
1. Push this repository to GitHub.
2. Create a new Web Service on Render.
3. Connect your GitHub repo.
4. Render will read render.yaml automatically, or set manually:
- Build command: pip install -r requirements.txt
- Start command: gunicorn app.main:app
5. Add environment variable in Render:
- OPENAI_API_KEY
6. Deploy.

## Security Notes
- Do not commit API keys.
- Keep OPENAI_API_KEY only in environment variables.
- uploads and local runtime files should not be committed.

## License
MIT

[![GitHub stars](https://img.shields.io/github/stars/Akshatageek/DocQuest.svg?style=social&label=Star)](https://github.com/Akshatageek/DocQuest)

---

**Made with ❤️ for educators and learners worldwide**

*Transform your documents into intelligent quizzes with DocQuest - where AI meets education!*
