"""Job Description Analyzer Module - Compares candidate profile with job description requirements."""
from typing import Dict, Any, List, Set
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class JDAnalyzer:
    """Analyzes candidate fit with job description."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI Job Description Analyzer. Analyze how well a candidate matches a job description.

CRITICAL INSTRUCTIONS FOR TECHNOLOGY MATCHING:
- Understand technology relationships and dependencies. If a candidate knows a technology, they likely know related tools:
  * Machine Learning (ML) → includes: scikit-learn, sklearn, numpy, pandas, scipy, matplotlib, seaborn, tensorflow, pytorch, keras
  * Deep Learning → includes: tensorflow, pytorch, keras, neural networks, CNNs, RNNs
  * Data Science → includes: pandas, numpy, matplotlib, seaborn, jupyter, data analysis
  * Web Development → includes: HTML, CSS, JavaScript, React, Vue, Angular, Node.js
  * Backend Development → includes: APIs, REST, databases, server-side frameworks
  * Cloud (AWS/Azure/GCP) → includes: EC2, S3, Lambda, storage, compute services
  * Python → includes: all Python libraries and frameworks
  * JavaScript → includes: Node.js, React, Vue, Angular, TypeScript
  * SQL → includes: MySQL, PostgreSQL, MongoDB, database management
  * DevOps → includes: Docker, Kubernetes, CI/CD, Git, Jenkins

- DO NOT penalize candidates for missing exact technology names if they have related/equivalent skills
- If JD mentions "scikit-learn" and candidate has "Machine Learning" or "sklearn", consider it a MATCH
- If JD mentions "pandas" and candidate has "Data Science" or "Python", consider it a MATCH
- Look for conceptual understanding, not just exact keyword matches
- Consider transferable skills and learning ability

Your task:
1. Extract required technologies and skills from the Job Description
2. Match candidate's skills, experience, and projects against JD requirements (using smart matching)
3. Calculate a relevance score (0-100) based on:
   - Technology/skill match (using smart matching, not exact keywords)
   - Educational background alignment
   - Experience relevance
   - Project relevance
   - Overall fit potential
4. Provide a concise AI suggestion (STRICTLY 2 lines maximum) explaining:
   - Key strengths and matches
   - Any significant gaps or overall assessment
   
IMPORTANT: Keep the AI Suggestion to exactly 2 lines. Be brief and direct.

Scoring Guidelines:
- 80-100: Excellent match - candidate has most/all required skills (including related/equivalent)
- 60-79: Good match - candidate has relevant skills, minor gaps
- 40-59: Moderate match - candidate has some relevant skills, some gaps
- 20-39: Weak match - candidate has limited relevant skills
- 0-19: Poor match - candidate lacks most required skills

Be generous with scoring when candidate has related/equivalent technologies. Focus on conceptual fit, not exact keyword matching."""),
            ("human", """Job Description:
{job_description}

Candidate Profile:
{candidate_profile}

Analyze the candidate's fit using smart technology matching and provide in this EXACT format:
1. JD Relevance Score (0-100): [number only, e.g., 85]
2. AI Suggestion: [exactly 2 lines, brief explanation]

CRITICAL: 
- The score must be a number between 0-100
- The suggestion must be exactly 2 lines
- Format: "JD Relevance Score (0-100): 85" followed by "AI Suggestion:" on next line""")
        ])

    def analyze(self, candidate_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Analyze candidate fit with job description.
        
        Args:
            candidate_data: Parsed candidate resume data
            job_description: Job description text
        
        Returns:
            Dictionary with score and suggestion
        """
        # Format candidate profile for analysis
        candidate_profile = self._format_candidate_profile(candidate_data)
        
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "job_description": job_description,
            "candidate_profile": candidate_profile
        })
        
        content = response.content.strip()
        
        # Extract suggestion first - get first 2 lines after "AI Suggestion:"
        suggestion = "Analysis completed."
        suggestion_patterns = [
            r'AI Suggestion[:\s]+(.+?)(?:\n\n|\n\n|$)',  # Until double newline or end
            r'AI Suggestion[:\s]+(.+?)(?:\n[^\n]*\n[^\n]*$)',  # Next 2 lines
            r'AI Suggestion[:\s]+(.+)',  # Everything after
        ]
        
        for pattern in suggestion_patterns:
            suggestion_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if suggestion_match:
                suggestion = suggestion_match.group(1).strip()
                # Limit to 2 lines
                lines = suggestion.split('\n')
                if len(lines) > 2:
                    suggestion = '\n'.join(lines[:2])
                break
        
        # Extract score - try multiple patterns
        score = 0
        score_patterns = [
            r'JD Relevance Score.*?(\d+)',  # Original pattern
            r'Score.*?(\d+)',  # Just "Score: 85"
            r'Relevance Score.*?(\d+)',  # "Relevance Score: 85"
            r'(\d+)\s*(?:out of 100|/100)',  # "85 out of 100" or "85/100"
            r'Score[:\s]+(\d+)',  # "Score: 85"
        ]
        
        for pattern in score_patterns:
            score_match = re.search(pattern, content, re.IGNORECASE)
            if score_match:
                try:
                    score = int(score_match.group(1))
                    if 0 <= score <= 100:
                        break
                except (ValueError, IndexError):
                    continue
        
        # If still no score found, try to find any number between 0-100 near "score" keyword
        if score == 0:
            score_context = re.search(r'score[^.]*?(\d{1,3})', content, re.IGNORECASE)
            if score_context:
                try:
                    potential_score = int(score_context.group(1))
                    if 0 <= potential_score <= 100:
                        score = potential_score
                except (ValueError, IndexError):
                    pass
        
        # Final fallback: Infer score from suggestion text if still 0
        if score == 0 and suggestion != "Analysis completed.":
            suggestion_lower = suggestion.lower()
            if any(word in suggestion_lower for word in ['excellent', 'perfect', 'ideal', 'strong match', 'highly qualified']):
                score = 85  # Default to high score if described as excellent
            elif any(word in suggestion_lower for word in ['good', 'strong', 'well', 'suitable', 'qualified']):
                score = 70  # Default to good score
            elif any(word in suggestion_lower for word in ['moderate', 'some', 'partial', 'adequate']):
                score = 50  # Default to moderate score
            elif any(word in suggestion_lower for word in ['weak', 'limited', 'lacks', 'missing']):
                score = 30  # Default to weak score
            else:
                score = 50  # Default moderate score if unclear
        
        # Clean up suggestion
        suggestion = re.sub(r'^\d+\.\s*', '', suggestion)  # Remove numbering
        suggestion = suggestion.strip()
        
        # If suggestion is too long, truncate to 2 sentences
        sentences = suggestion.split('. ')
        if len(sentences) > 2:
            suggestion = '. '.join(sentences[:2])
            if not suggestion.endswith('.'):
                suggestion += '.'
        
        return {
            "jd_relevance_score": min(100, max(0, score)),
            "ai_suggestion": suggestion
        }

    def _format_candidate_profile(self, candidate_data: Dict[str, Any]) -> str:
        """Format candidate data for JD analysis."""
        profile_parts = []
        
        # Education
        education = candidate_data.get("Education", [])
        if education:
            edu_str = "Education: "
            edu_parts = []
            for edu in education:
                parts = []
                if edu.get("Degree"):
                    parts.append(edu["Degree"])
                if edu.get("Specialization"):
                    parts.append(f"in {edu['Specialization']}")
                if edu.get("College"):
                    parts.append(f"from {edu['College']}")
                if parts:
                    edu_parts.append(", ".join(parts))
            if edu_parts:
                profile_parts.append(edu_str + "; ".join(edu_parts))
        
        # Skills
        skills = candidate_data.get("Skills", [])
        if skills and isinstance(skills, list):
            profile_parts.append(f"Skills: {', '.join(skills[:20])}")  # Limit to first 20
        
        # Experience
        experience = candidate_data.get("Experience", [])
        if experience and isinstance(experience, list):
            exp_parts = []
            for exp in experience:
                parts = []
                if exp.get("Role"):
                    parts.append(exp["Role"])
                if exp.get("Company"):
                    parts.append(f"at {exp['Company']}")
                if exp.get("Duration"):
                    parts.append(f"({exp['Duration']})")
                if parts:
                    exp_parts.append(" ".join(parts))
            if exp_parts:
                profile_parts.append("Experience: " + "; ".join(exp_parts))
        
        # Projects
        projects = candidate_data.get("Projects", [])
        if projects and isinstance(projects, list):
            proj_tech = []
            for proj in projects[:5]:  # Limit to first 5
                tech = proj.get("TechStack", "")
                if tech:
                    proj_tech.append(tech)
            if proj_tech:
                profile_parts.append(f"Technologies Used: {', '.join(set(', '.join(proj_tech).split(', ')))}")
        
        return "\n".join(profile_parts)

