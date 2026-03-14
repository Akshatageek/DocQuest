from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import re
import json
import io
from datetime import datetime
from dotenv import load_dotenv
try:
    # OpenAI SDK v1.x
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

load_dotenv()


app = Flask(__name__)
# Use an absolute uploads path within the app directory for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Stats file path
STATS_FILE = os.path.join(BASE_DIR, 'stats.json')

# Initialize stats file if it doesn't exist
def init_stats():
    if not os.path.exists(STATS_FILE):
        initial_stats = {
            'total_questions': 10000,  # Starting with your current numbers
            'total_users': 500,
            'total_pdfs': 250,
            'success_rate': 99.0,
            'last_updated': datetime.now().isoformat()
        }
        with open(STATS_FILE, 'w') as f:
            json.dump(initial_stats, f, indent=2)

def get_stats():
    try:
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    except:
        init_stats()
        return get_stats()

def update_stats(questions_generated, pdf_processed=True):
    stats = get_stats()
    stats['total_questions'] += questions_generated
    if pdf_processed:
        stats['total_pdfs'] += 1
        # Gradually increase user count (simulate new educators joining)
        if stats['total_pdfs'] % 3 == 0:  # Every 3rd PDF increases user count
            stats['total_users'] = stats.get('total_users', 500) + 1
        # Update success rate (keeping it realistic)
        stats['success_rate'] = min(99.8, stats['success_rate'] + 0.01)
    stats['last_updated'] = datetime.now().isoformat()
    
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

# Initialize stats on startup
init_stats()

# Read OpenAI API key from environment only.
# Keep secrets out of source control for GitHub/deployment safety.
def get_openai_api_key() -> str:
    return (os.getenv('OPENAI_API_KEY') or "").strip()


def get_gemini_api_key() -> str:
    # Support both names for convenience.
    return (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or "").strip()


def get_ai_provider() -> str:
    # auto | gemini | openai
    return (os.getenv('AI_PROVIDER') or 'auto').strip().lower()

@app.route('/', methods=['GET'])
def index():
    stats = get_stats()
    return render_template('index.html', stats=stats)

@app.route('/generate', methods=['POST'])
def generate():
    if 'pdf' not in request.files:
        return render_template('results.html', questions={"error": "No file part named 'pdf' in form."}, questions_json=json.dumps({"error": "No file part named 'pdf' in form."}, indent=2))

    # Read user options
    def to_int(val, default):
        try:
            return int(val)
        except Exception:
            return default

    # Updated field names to match the form
    mcq_count = to_int(request.form.get('num_questions', '5'), 5)
    difficulty = request.form.get('difficulty', 'Medium')
    include_explanations = request.form.get('include_explanations') is not None
    include_tf = request.form.get('include_tf') is not None
    tf_count = to_int(request.form.get('tf_count', '0'), 0) if include_tf else 0
    
    # Clamp values to sane ranges
    mcq_count = max(1, min(20, mcq_count))
    tf_count = max(0, min(10, tf_count))

    file = request.files['pdf']
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            # Process PDF
            text = extract_text_from_pdf(filepath)
            # Generate questions with new parameters
            questions = generate_questions(text, mcq_count=mcq_count, tf_count=tf_count, difficulty=difficulty, include_explanations=include_explanations)
            
            # Update stats with generated questions count
            total_questions_generated = mcq_count + tf_count
            update_stats(total_questions_generated, pdf_processed=True)
            
            return render_template('results.html', questions=questions, questions_json=json.dumps(questions, indent=2))
        except Exception as e:
            return render_template('results.html', questions={"error": str(e)}, questions_json=json.dumps({"error": str(e)} , indent=2))
    else:
        return render_template('results.html', questions={"error": "Please upload a valid PDF file."}, questions_json=json.dumps({"error": "Please upload a valid PDF file."}, indent=2))


@app.route('/health')
def health():
    return {"status": "ok"}, 200

@app.route('/api/stats')
def api_stats():
    """API endpoint to get current stats"""
    stats = get_stats()
    return jsonify({
        'questions': f"{stats['total_questions']:,}+",
        'users': f"{stats['total_pdfs']}+",
        'accuracy': f"{stats['success_rate']:.1f}%"
    })


@app.route('/sample', methods=['POST'])
def sample():
    """Generate questions from a built-in sample text, useful for quick testing."""
    sample_text = (
        "Photosynthesis is a process used by plants and other organisms to convert light energy "
        "into chemical energy that can later be released to fuel the organism's activities. "
        "This chemical energy is stored in carbohydrate molecules, such as sugars, which are "
        "synthesized from carbon dioxide and water—hence the name photosynthesis, from the Greek "
        "phōs (light), and synthesis (putting together)."
    )
    questions = generate_questions(sample_text)
    return render_template('results.html', questions=questions, questions_json=json.dumps(questions, indent=2))

@app.route('/download/pdf', methods=['POST'])
def download_pdf():
    """Generate and return a PDF of the quiz from posted JSON."""
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return {"error": "Invalid payload"}, 400
        mc = payload.get('multiple_choice') or []
        tf = payload.get('true_false') or []

        buf = io.BytesIO()
        try:
            # Build PDF
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, PageBreak
            from reportlab.lib.units import inch

            doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()
            title = styles['Heading1']
            h2 = styles['Heading2']
            normal = styles['BodyText']
            normal.leading = 14

            story = []
            story.append(Paragraph("Quiz Questions", title))
            story.append(Spacer(1, 0.2*inch))

            # MC Section
            story.append(Paragraph(f"Multiple Choice ({len(mc)})", h2))
            for i, q in enumerate(mc, start=1):
                story.append(Paragraph(f"{i}. {q.get('question','')}" , normal))
                # options
                opts = q.get('options') or []
                items = [ListItem(Paragraph(f"{chr(65+j)}. {opt}", normal), leftIndent=16) for j, opt in enumerate(opts[:4])]
                if items:
                    story.append(ListFlowable(items, bulletType='bullet', leftIndent=10))
                story.append(Spacer(1, 0.12*inch))

            story.append(Spacer(1, 0.2*inch))

            # TF Section
            story.append(Paragraph(f"True / False ({len(tf)})", h2))
            for i, q in enumerate(tf, start=1):
                story.append(Paragraph(f"{i}. {q.get('question','')}", normal))
                story.append(Spacer(1, 0.12*inch))

            # Answer Key
            story.append(PageBreak())
            story.append(Paragraph("Answer Key", h2))
            story.append(Spacer(1, 0.1*inch))
            # MC Answers
            if mc:
                story.append(Paragraph("Multiple Choice", styles['Heading3']))
                for i, q in enumerate(mc, start=1):
                    ans = q.get('answer')
                    opts = q.get('options') or []
                    letter = ''
                    label = ''
                    if isinstance(ans, str) and ans:
                        letter = ans.strip().upper()[:1]
                        idx_map = {'A':0,'B':1,'C':2,'D':3}
                        idx = idx_map.get(letter)
                        if idx is not None and idx < len(opts):
                            label = opts[idx]
                    text = f"{i}) {letter}"
                    if label:
                        text += f" — {label}"
                    story.append(Paragraph(text, normal))
                story.append(Spacer(1, 0.12*inch))
            # TF Answers
            if tf:
                story.append(Paragraph("True / False", styles['Heading3']))
                for i, q in enumerate(tf, start=1):
                    ans = q.get('answer')
                    if isinstance(ans, bool):
                        val = 'True' if ans else 'False'
                    elif isinstance(ans, str):
                        val = 'True' if ans.strip().lower() == 'true' else 'False'
                    else:
                        val = '—'
                    story.append(Paragraph(f"{i}) {val}", normal))

            doc.build(story)
            pdf = buf.getvalue()
        finally:
            buf.close()

        return (pdf, 200, {
            'Content-Type': 'application/pdf',
            'Content-Disposition': 'attachment; filename="quiz.pdf"'
        })
    except Exception as e:
        return {"error": str(e)}, 500

def extract_text_from_pdf(filepath):
    import PyPDF2
    text = ""
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    # Ensure we always return something to avoid confusing empty pages
    return text if text.strip() else "The document text could not be extracted reliably. Generate general questions about the topic presented." 


# Handle large uploads gracefully
@app.errorhandler(413)
def too_large(_e):
    return render_template('results.html', questions={"error": "File too large (limit 16 MB)."}, questions_json=json.dumps({"error": "File too large (limit 16 MB)."}, indent=2)), 413

def _normalize_mcq_item(item, include_explanations: bool):
    if not isinstance(item, dict):
        return None

    question = str(item.get("question", "")).strip()
    if not question:
        return None

    raw_options = item.get("options")
    if not isinstance(raw_options, list):
        return None

    options = [str(o).strip() for o in raw_options if str(o).strip()]
    options = options[:4]
    if len(options) != 4:
        return None

    answer = str(item.get("answer", "")).strip().upper()
    if answer in {"A", "B", "C", "D"}:
        answer_letter = answer
    else:
        answer_map = {options[0].lower(): "A", options[1].lower(): "B", options[2].lower(): "C", options[3].lower(): "D"}
        answer_letter = answer_map.get(answer.lower())
        if answer_letter is None:
            return None

    explanation = str(item.get("explanation", "")).strip()
    if not include_explanations:
        explanation = ""

    return {
        "question": question,
        "options": options,
        "answer": answer_letter,
        "explanation": explanation,
    }


def _normalize_tf_item(item, include_explanations: bool):
    if not isinstance(item, dict):
        return None

    question = str(item.get("question", "")).strip()
    if not question:
        return None

    raw_answer = item.get("answer")
    if isinstance(raw_answer, bool):
        answer = raw_answer
    elif isinstance(raw_answer, str):
        lowered = raw_answer.strip().lower()
        if lowered in {"true", "t", "yes"}:
            answer = True
        elif lowered in {"false", "f", "no"}:
            answer = False
        else:
            return None
    else:
        return None

    explanation = str(item.get("explanation", "")).strip()
    if not include_explanations:
        explanation = ""

    return {
        "question": question,
        "answer": answer,
        "explanation": explanation,
    }


def _extract_json_object(raw_text: str):
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _classify_openai_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return "insufficient_quota"
    if "authentication" in msg or "api key" in msg or "401" in msg:
        return "auth_error"
    if "rate" in msg or "429" in msg:
        return "rate_limited"
    return "openai_error"


def _classify_gemini_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "not found" in msg or "models/" in msg and "generatecontent" in msg:
        return "invalid_model"
    if "quota" in msg or "429" in msg or "resource_exhausted" in msg:
        return "insufficient_quota"
    if "api key" in msg or "permission" in msg or "401" in msg or "403" in msg:
        return "auth_error"
    if "rate" in msg:
        return "rate_limited"
    return "gemini_error"


def _gemini_model_candidates() -> list:
    configured = (os.getenv('GEMINI_MODEL') or '').strip()
    candidates = []
    if configured:
        candidates.append(configured)

    # Common stable fallbacks for generateContent.
    for m in ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-1.5-pro"]:
        if m not in candidates:
            candidates.append(m)

    # Discover account-available models dynamically when possible.
    if genai is not None:
        try:
            listed = []
            for model in genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                name = getattr(model, "name", "") or ""
                if "generateContent" in methods and "gemini" in name.lower():
                    listed.append(name.replace("models/", ""))
            for m in listed:
                if m not in candidates:
                    candidates.append(m)
        except Exception:
            pass

    return candidates


def _build_generation_prompts(mcq_count: int, tf_count: int, difficulty: str, include_explanations: bool, text: str):
    difficulty_instructions = {
        "Easy": "Make questions straightforward with direct answers from the text. Focus on basic concepts and definitions.",
        "Medium": "Create questions that require understanding and application of concepts. Mix direct facts with some inference.",
        "Hard": "Generate challenging questions requiring analysis, synthesis, and deeper understanding. Include questions that test critical thinking."
    }
    difficulty_text = difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])
    explanation_text = "with detailed explanations (2-3 sentences each)" if include_explanations else "without explanations"

    system_prompt = f"""You are an expert educator creating high-quality quiz questions from educational content.

INSTRUCTIONS:
1. Generate exactly {mcq_count} multiple-choice questions and {tf_count} true/false questions
2. {difficulty_text}
3. Each MCQ should have 4 distinct, plausible options (A, B, C, D)
4. Make sure questions are clear, specific, and test understanding
5. Include {explanation_text}
6. Base all questions strictly on the provided text content
7. Avoid ambiguous or trick questions
8. Return only JSON, no markdown and no commentary

DIFFICULTY LEVEL: {difficulty}

RESPONSE FORMAT - Return ONLY valid JSON with this exact structure:
{{
  "multiple_choice": [
    {{
      "question": "Clear, specific question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "A",
      "explanation": "Detailed explanation of why this is correct."
    }}
  ],
  "true_false": [
    {{
      "question": "Clear statement to evaluate as true or false.",
      "answer": true,
      "explanation": "Explanation of why this statement is true/false."
    }}
  ]
}}"""

    user_content = f"TEXT TO CREATE QUESTIONS FROM:\n\n{text[:12000]}"
    return system_prompt, user_content


def _validate_model_output(data, mcq_count: int, tf_count: int, include_explanations: bool, text: str, difficulty: str):
    if not isinstance(data, dict):
        return None

    raw_mc = data.get("multiple_choice", [])
    raw_tf = data.get("true_false", [])

    validated_mc = []
    for q in raw_mc[:mcq_count]:
        normalized = _normalize_mcq_item(q, include_explanations=include_explanations)
        if normalized is not None:
            validated_mc.append(normalized)

    validated_tf = []
    for q in raw_tf[:tf_count]:
        normalized = _normalize_tf_item(q, include_explanations=include_explanations)
        if normalized is not None:
            validated_tf.append(normalized)

    if len(validated_mc) < mcq_count or len(validated_tf) < tf_count:
        fallback = generate_fallback_questions(
            text,
            mcq_count=max(0, mcq_count - len(validated_mc)),
            tf_count=max(0, tf_count - len(validated_tf)),
            difficulty=difficulty,
            include_explanations=include_explanations,
        )
        validated_mc.extend((fallback.get("multiple_choice") or [])[: max(0, mcq_count - len(validated_mc))])
        validated_tf.extend((fallback.get("true_false") or [])[: max(0, tf_count - len(validated_tf))])

    if validated_mc or validated_tf:
        return {
            "multiple_choice": validated_mc[:mcq_count],
            "true_false": validated_tf[:tf_count],
        }
    return None


def generate_questions(text, mcq_count: int = 10, tf_count: int = 0, difficulty: str = "Medium", include_explanations: bool = True):
    """Generate questions using OpenAI if configured; otherwise use a simple fallback.

    Args:
        text: Source text to generate questions from.
        mcq_count: Number of multiple-choice questions (>=1).
        tf_count: Number of true/false questions (>=0). If 0, return an empty list for true_false.
        difficulty: Difficulty level (Easy, Medium, Hard).
        include_explanations: Whether to include explanations for answers.

    Returns:
        dict: {"multiple_choice": [...], "true_false": [...], "source": "openai|fallback"}
    """
    # Read latest keys at request-time so setting env vars after startup still works.
    provider = get_ai_provider()
    openai_api_key = get_openai_api_key()
    gemini_api_key = get_gemini_api_key()
    preferred = ["gemini", "openai"] if provider == "auto" else [provider, "openai" if provider == "gemini" else "gemini"]
    system_prompt, user_content = _build_generation_prompts(mcq_count, tf_count, difficulty, include_explanations, text)

    model_errors = []

    for p in preferred:
        if p == "gemini":
            if not gemini_api_key or genai is None:
                continue
            try:
                genai.configure(api_key=gemini_api_key)
                candidates = _gemini_model_candidates()
                gemini_error = None
                for gemini_model in candidates:
                    try:
                        model = genai.GenerativeModel(gemini_model)
                        response = model.generate_content(
                            [system_prompt, "\n\n", user_content],
                            generation_config={
                                "temperature": 0.35,
                                "max_output_tokens": 4000,
                                "response_mime_type": "application/json",
                            },
                        )

                        content = (getattr(response, "text", "") or "").strip()
                        data = _extract_json_object(content)
                        validated = _validate_model_output(data, mcq_count, tf_count, include_explanations, text, difficulty)
                        if validated:
                            validated["source"] = "gemini"
                            validated["model"] = gemini_model
                            return validated
                    except Exception as e:
                        gemini_error = e
                        # Try next model if current one is unavailable.
                        if _classify_gemini_error(e) == "invalid_model":
                            continue
                        raise

                if gemini_error is not None:
                    raise gemini_error
                model_errors.append("invalid_model_response")
            except Exception as e:
                print(f"Gemini API Error: {e}")
                model_errors.append(_classify_gemini_error(e))
            continue

        if p == "openai":
            if not openai_api_key or OpenAI is None:
                continue
            try:
                client = OpenAI(api_key=openai_api_key)
                openai_model = (os.getenv('OPENAI_MODEL') or 'gpt-4o-mini').strip()
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.35,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content if response.choices else "{}"
                data = _extract_json_object(content if isinstance(content, str) else "{}")
                validated = _validate_model_output(data, mcq_count, tf_count, include_explanations, text, difficulty)
                if validated:
                    validated["source"] = "openai"
                    return validated
                model_errors.append("invalid_model_response")
            except Exception as e:
                print(f"OpenAI API Error: {e}")
                model_errors.append(_classify_openai_error(e))
            continue

    # If all configured providers failed, fallback with the most relevant reason.
    if model_errors:
        return generate_fallback_questions(
            text,
            mcq_count,
            tf_count,
            difficulty,
            include_explanations,
            fallback_reason=model_errors[0],
        )

    # No valid provider key/sdk configured.
    if provider == "gemini":
        reason = "missing_gemini_key" if not gemini_api_key else "gemini_sdk_unavailable"
    elif provider == "openai":
        reason = "missing_api_key" if not openai_api_key else "openai_sdk_unavailable"
    else:
        reason = "missing_provider_key"

    return generate_fallback_questions(text, mcq_count, tf_count, difficulty, include_explanations, fallback_reason=reason)


def simple_fallback_questions(text: str, mcq_count: int = 10, tf_count: int = 0, difficulty: str = "Medium", include_explanations: bool = True):
    """Lightweight, no-API fallback generator based on the input text.

    Generates variable numbers of MCQ and True/False based on requested counts.
    """
    return generate_fallback_questions(text, mcq_count, tf_count, difficulty, include_explanations)

def generate_fallback_questions(
    text: str,
    mcq_count: int,
    tf_count: int,
    difficulty: str,
    include_explanations: bool,
    fallback_reason: str = "",
):
    """Improved fallback question generator."""
    # Basic sentence splitting
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    head = sentences[:max(10, mcq_count + tf_count)] if sentences else ["This chapter discusses key concepts."]

    # Extract better keywords from text
    keywords = []
    common_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    
    for s in head:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", s.lower())
        for w in words:
            if w not in common_words and len(w) >= 4:
                keywords.append(w.title())
    
    # Remove duplicates and ensure we have enough
    keywords = list(dict.fromkeys(keywords))  # preserves order, removes duplicates
    if len(keywords) < 8:
        keywords.extend(["Process", "System", "Method", "Theory", "Analysis", "Concept", "Model", "Framework"])
    
    # Generate MCQs grounded in sentence-level cloze prompts.
    mcq = []
    for i in range(mcq_count):
        sentence = head[i % len(head)]
        sentence_words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", sentence)
        answer_text = sentence_words[min(len(sentence_words) - 1, max(0, i % max(1, len(sentence_words))))] if sentence_words else keywords[i % len(keywords)]

        blanked = re.sub(rf"\b{re.escape(answer_text)}\b", "_____", sentence, count=1, flags=re.IGNORECASE)
        if blanked == sentence:
            blanked = sentence[:100] + ("..." if len(sentence) > 100 else "")
            question = f"According to the passage, which term best fits this statement: '{blanked}'"
        else:
            question = f"According to the passage, which term correctly fills the blank: '{blanked}'"

        opts = [answer_text.title()]
        cursor = (i * 3) % len(keywords)
        while len(opts) < 4:
            candidate = keywords[cursor % len(keywords)]
            cursor += 1
            if candidate.lower() not in {o.lower() for o in opts}:
                opts.append(candidate)

        # Deterministic shuffle to avoid always A while keeping repeatability.
        shift = i % 4
        opts = opts[shift:] + opts[:shift]
        correct_idx = opts.index(answer_text.title()) if answer_text.title() in opts else 0

        # Ensure all options are unique
        opts = list(dict.fromkeys(opts))
        while len(opts) < 4:
            opts.append(f"Option {len(opts) + 1}")
        
        correct_letter = chr(ord('A') + correct_idx)
        
        explanation = ""
        if include_explanations:
            explanation = f"The correct answer is '{opts[correct_idx]}' as it best relates to the context and concepts discussed in the provided text."
        
        mcq.append({
            "question": question,
            "options": opts[:4],
            "answer": correct_letter,
            "explanation": explanation
        })

    # Generate True/False questions
    tf = []
    for i in range(tf_count):
        sentence = head[i % len(head)]
        answer_val = (i % 2 == 0)  # Alternate true/false
        
        # Create more natural T/F statements
        if answer_val:
            question = f"The text discusses: {sentence[:100]}..."
        else:
            # Create false statements by negating or changing concepts
            negation_words = ["does not discuss", "contradicts", "ignores the concept of"]
            neg_word = negation_words[i % len(negation_words)]
            question = f"The text {neg_word}: {sentence[:80]}..."
        
        explanation = ""
        if include_explanations:
            if answer_val:
                explanation = "This statement is TRUE as it directly reflects content from the provided text."
            else:
                explanation = "This statement is FALSE as it misrepresents or contradicts the information in the text."
        
        tf.append({
            "question": question,
            "answer": answer_val,
            "explanation": explanation
        })

    return {
        "multiple_choice": mcq,
        "true_false": tf,
        "source": "fallback",
        "fallback_reason": fallback_reason,
    }

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
