"""Standalone Resume Parser - Extracts structured information from raw resume text and returns JSON according to the required schema."""
import json
import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import PyPDF2
import io


class ResumeParser:
    """Parses raw resume text into structured JSON format."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an AI Resume Parser. You will receive raw text extracted from a PDF resume. Your task is to extract all structured information and return ONLY valid JSON according to the schema described below.

CRITICAL JSON FORMATTING RULES:
- Output ONLY valid JSON with NO markdown code fences (no ```json or ```)
- Use double quotes for all strings, NOT single quotes
- NO trailing commas before closing braces or brackets
- NO comments in the JSON
- Ensure all brackets and braces are properly closed

Data Extraction Rules:
- Missing fields must be returned as "info not available in resume"
- Extract exact information from the resume, do not invent data

Education:
Return as a list of objects. Each object must have:
- College: Name of the college/university
- Degree: Degree obtained (B.Tech, M.Tech, MBA, etc.)
- Specialization: Field of study
- Grade: Marks, percentage, or CGPA

School:
Return as a list of strings with all schools attended.

Experience:
Return as a list of objects. Each object must include:
- Company: Company name
- Role: Job title
- Duration: Total work duration (calculate if dates provided, e.g., Jan 2020 – Feb 2023 = 3 years 1 month)
- Description: A short paragraph describing responsibilities, work done, and client names if available

Projects:
Return as a list of objects. Each object must include:
- ProjectName: Name of the project
- TechStack: Technologies used
- Description: Short description of the project

Skills:
Return as a list of technical skills (programming languages, frameworks, tools, databases, cloud platforms, etc.)

Certifications, Activities_Hobbies, Achievements, Languages:
Return as lists of strings.

State, City, Country:
Return State and City where the candidate studied or lives.
If Country is missing, infer from State/City; if not possible, return "info not available in resume".

Name, Number, Email:
Extract full name, phone number, and email address. Return "info not available in resume" if missing.

Output formatting:
- Return valid JSON with "output" as the root key
- Use lists of objects for multi-entry fields (Education, Experience, Projects)
- Use lists of strings for simple multi-entry fields (School, Skills, Certifications, Languages, Achievements, Activities_Hobbies)

Return ONLY the JSON object. Example structure:
{{"output": {{"Name": "...", "Email": "...", "Skills": ["...", "..."], "Education": [{{"College": "...", "Degree": "...", "Specialization": "...", "Grade": "..."}}]}}}}""",
            ),
            ("human", "Resume Text:\n{resume_text}"),
        ])

    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """Extract text from PDF file bytes."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues from LLM responses."""
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix common quote issues - but be careful with apostrophes in content
        # This is a simple approach that may need refinement
        
        return json_str.strip()
    
    def parse(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse raw resume text into structured JSON and normalize to the schema.
        Args:
            resume_text: Raw text extracted from resume PDF
        Returns:
            Parsed resume data as a dictionary (value of the 'output' key)
        """
        chain = self.prompt_template | self.llm
        response = chain.invoke({"resume_text": resume_text})
        
        # Extract JSON from response
        content = response.content.strip()
        
        # Remove markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Handle both ```json and ``` cases
            if lines[0].lower().startswith("```json") or lines[0] == "```":
                content = "\n".join(lines[1:])
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        # Try parsing multiple strategies
        parsing_attempts = []
        
        # Attempt 1: Direct parse
        try:
            parsed_data = json.loads(content)
            data = parsed_data.get("output", parsed_data)
            return self._normalize_output(data)
        except json.JSONDecodeError as e1:
            parsing_attempts.append(f"Direct parse failed: {str(e1)}")
        
        # Attempt 2: Clean and parse
        try:
            cleaned_content = self._clean_json_string(content)
            parsed_data = json.loads(cleaned_content)
            data = parsed_data.get("output", parsed_data)
            return self._normalize_output(data)
        except json.JSONDecodeError as e2:
            parsing_attempts.append(f"Cleaned parse failed: {str(e2)}")
        
        # Attempt 3: Extract JSON object with regex
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                json_str = self._clean_json_string(json_str)
                parsed_data = json.loads(json_str)
                data = parsed_data.get("output", parsed_data)
                return self._normalize_output(data)
            except json.JSONDecodeError as e3:
                parsing_attempts.append(f"Regex extraction failed: {str(e3)}")
        
        # Attempt 4: Try to extract just the "output" object
        output_pattern = r'"output"\s*:\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}'
        output_match = re.search(output_pattern, content, re.DOTALL)
        if output_match:
            try:
                # Wrap in braces to make valid JSON
                json_str = "{" + output_match.group() + "}"
                json_str = self._clean_json_string(json_str)
                parsed_data = json.loads(json_str)
                data = parsed_data.get("output", {})
                return self._normalize_output(data)
            except json.JSONDecodeError as e4:
                parsing_attempts.append(f"Output extraction failed: {str(e4)}")
        
        # If all attempts fail, log details and raise error
        print("❌ All JSON parsing attempts failed:")
        for i, attempt in enumerate(parsing_attempts, 1):
            print(f"   Attempt {i}: {attempt}")
        print(f"📄 LLM Response (first 500 chars): {content[:500]}")
        
        # Return error with context
        raise ValueError(
            f"Failed to parse JSON from LLM response after {len(parsing_attempts)} attempts. "
            f"Last error: {parsing_attempts[-1] if parsing_attempts else 'Unknown'}"
        )

    # ----------------------- normalization helpers ----------------------- #
    
    @staticmethod
    def _dedup_list(items):
        seen = set()
        deduped = []
        for item in items:
            key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _normalize_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all fields exist, are properly typed, de-duplicated, and have 'info not available in resume' where missing."""
        default_str = "info not available in resume"
        
        # Simple string fields
        result = {
            "Name": data.get("Name", default_str),
            "Number": data.get("Number", default_str),
            "Email": data.get("Email", default_str),
            "State": data.get("State", default_str),
            "City": data.get("City", default_str),
            "Country": data.get("Country", default_str),
        }
        
        # List of strings fields
        list_string_fields = ["School", "Skills", "Certifications", "Activities_Hobbies", "Achievements", "Languages"]
        for field in list_string_fields:
            value = data.get(field, [])
            if not isinstance(value, list):
                value = []
            value = [v for v in value if v and v != default_str]
            value = self._dedup_list(value)
            if not value:
                value = [default_str]
            result[field] = value
        
        # List of object fields with required keys
        def normalize_obj_list(items, keys):
            if not isinstance(items, list):
                items = []
            normalized = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                norm_item = {k: item.get(k, default_str) if item.get(k) else default_str for k in keys}
                normalized.append(norm_item)
            if not normalized:
                normalized = [{k: default_str for k in keys}]
            return self._dedup_list(normalized)
        
        result["Education"] = normalize_obj_list(
            data.get("Education", []),
            ["College", "Degree", "Specialization", "Grade"],
        )
        result["Projects"] = normalize_obj_list(
            data.get("Projects", []),
            ["ProjectName", "TechStack", "Description"],
        )
        result["Experience"] = normalize_obj_list(
            data.get("Experience", []),
            ["Company", "Role", "Duration", "Description"],
        )
        
        return result

