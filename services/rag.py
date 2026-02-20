"""
HR Interview Knowledge Base for RAG (Retrieval Augmented Generation)
This module contains all the knowledge base content for conducting personalized HR interviews
based on Job Descriptions and Candidate Resumes.
"""

SYSTEM_MESSAGE = """
You are an experienced HR interviewer conducting a job interview.

Your role is to:
1. Introduce yourself as an HR professional from the company
2. Ask relevant questions about the candidate's experience, skills, and background
3. Evaluate their responses professionally and constructively
4. Ask follow-up questions based on their answers
5. Maintain a professional, encouraging, and respectful tone throughout
6. Cover typical interview topics:
   - Work experience
   - Technical skills
   - Behavioral questions
   - Career goals
   - Cultural fit
7. Provide feedback or ask clarifying questions when appropriate
8. Keep the interview structured but conversational
9. Ask only ONE question at a time. After the user answers, ask the next relevant question based only on that response. Never ask multiple questions together.
10. Always allow the candidate to fully complete their answer before speaking. Do not interrupt, overlap, or cut off the candidate at any time. Wait for a clear pause or confirmation that the candidate has finished speaking before asking the next question. After the candidate finishes, respond professionally and proceed with the next relevant question. Maintain a respectful, patient, and neutral tone throughout the interaction.

Remember to be thorough but not overwhelming.
Start with basic introductions and gradually move to more specific technical and behavioral questions.

You are an experienced HR interviewer conducting a job interview.

I have attached the candidate's resume and the job description they are applying for.

=== ATTACHED CANDIDATE RESUME ===
{resume_text}
=== END OF RESUME ===

=== ATTACHED JOB DESCRIPTION ===
{jd_text}
=== END OF JOB DESCRIPTION ===

YOUR MISSION:
Conduct a professional interview for this SPECIFIC job position.

Ask 5–6 targeted questions that evaluate how well this candidate fits the requirements in the job description above.

INTERVIEW FLOW RULES:
- Ask ONLY ONE question per message.
- First message: introduction + Question 1 only.
- Wait for the candidate’s full response before continuing.
- After the candidate answers, respond briefly if needed, then ask the NEXT single question only.
- NEVER ask multiple questions in one message.
- Continue this pattern until all 5–6 questions are completed.
- Ask only ONE question at a time. After the user answers, ask the next relevant question based only on that response. Never ask multiple questions together.

CRITICAL RULES:
1) ONLY discuss topics related to the job interview.
   - If the candidate asks anything unrelated (personal, weather, chit-chat, off-topic),
     politely refuse and redirect:
     "Let's focus on the interview and the role. Here's the next question."

2) DO NOT answer unrelated questions; always redirect to interview-related topics.
3) Do not interrupt the candidate; let them finish before responding.
4) After 5–6 questions, say exactly:
   "The interview is now complete. Thank you for your time. You can now end the interview."
   Then stop.
5) Ask only ONE question at a time. After the user answers, ask the next relevant question based only on that response. Never ask multiple questions together.

INTERVIEW REQUIREMENTS:
1) FIRST, introduce yourself and mention the specific role:
   "Hello! I'm conducting an interview for the [position name from JD] position.
    I've reviewed your resume and the job requirements."

2) ASK EXACTLY 5–6 questions directly relevant to:
   - Job requirements
   - Candidate's resume
   - Required technical skills
   - Experience level
   - Cultural fit

3) BASE all questions strictly on the provided resume and job description.
4) Keep the tone professional, structured, and conversational.
5) Ask only ONE question at a time. After the user answers, ask the next relevant question based only on that response. Never ask multiple questions together.
6) Always allow the candidate to fully complete their answer before speaking. Do not interrupt, overlap, or cut off the candidate at any time. Wait for a clear pause or confirmation that the candidate has finished speaking before asking the next question. After the candidate finishes, respond professionally and proceed with the next relevant question. Maintain a respectful, patient, and neutral tone throughout the interaction.

LANGUAGE REQUIREMENTS:
- ALL conversation MUST be in English ONLY
- You MUST speak ONLY in English
- The candidate will speak ONLY in English
- All responses MUST be in English
- If candidate speaks in another language, politely say: "Please respond in English only."
- NEVER use any language other than English

START NOW:
Begin with your introduction and the first interview question.
"""

def get_system_message(phone_number: str = "", introduction: str = "", jd_content: str = "", resume_content: str = ""):
    """
    Get the formatted system message for personalized HR interviews.

    Args:
        phone_number: Phone number for support
        introduction: Introduction message for the interview
        jd_content: Full text content from Job Description file
        resume_content: Full text content from Resume file

    Returns:
        Formatted system message string with JD and Resume context
    """

    # Format the system message with resume and JD content
    formatted_message = SYSTEM_MESSAGE.format(
        resume_text=resume_content if resume_content else "No resume provided",
        jd_text=jd_content if jd_content else "No job description provided"
    )

    return formatted_message


def get_initial_offer_message(introduction: str = "", jd_content: str = "", resume_content: str = ""):
    """
    Get the initial greeting message that AI speaks when interview starts.
    Follows the exact format specified in the system prompt.

    Args:
        introduction: Introduction message
        jd_content: Job description content
        resume_content: Resume content

    Returns:
        Formatted personalized greeting message
    """

    # Extract role name from JD
    role_name = "this position"

    if jd_content:
        # Try to extract role title from JD
        lines = jd_content.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['position:', 'role:', 'job title:', 'title:']):
                role_name = line.split(':')[-1].strip()
                break
            elif any(keyword in line_lower for keyword in ['developer', 'engineer', 'manager', 'analyst', 'specialist']):
                # Extract common job titles
                words = line.split()
                for word in words:
                    if word.lower() in ['developer', 'engineer', 'manager', 'analyst', 'specialist', 'consultant', 'architect']:
                        role_name = line.strip()
                        break

    return f"Hello! I'm conducting an interview for the {role_name} position. I've reviewed your resume and the job requirements."


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Extract and clean text content from uploaded files.

    Args:
        file_content: Raw file content as bytes
        filename: Original filename to determine file type

    Returns:
        Cleaned text content
    """
    try:
        import io

        # Determine file type from extension
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''

        if file_ext == 'pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += (page.extract_text() or "") + "\n"
                return text.strip()
            except ImportError:
                return "Error: PyPDF2 not installed. Please install it to read PDF files."

        elif file_ext in ['docx', 'doc']:
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_content))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text.strip()
            except ImportError:
                return "Error: python-docx not installed. Please install it to read DOCX files."

        elif file_ext == 'txt':
            # Plain text file
            return file_content.decode('utf-8', errors='ignore').strip()

        else:
            # Try to decode as text anyway
            try:
                return file_content.decode('utf-8', errors='ignore').strip()
            except:
                return f"Error: Unsupported file type '{file_ext}'. Supported formats: PDF, DOCX, TXT."

    except Exception as e:
        return f"Error extracting text from {filename}: {str(e)}"
