"""Output Formatter Module - Formats the comparison results into the required output format."""
from typing import Dict, Any, List
from datetime import datetime


class OutputFormatter:
    """Formats comparison results into structured output."""
    
    def format(
        self,
        candidate_data: Dict[str, Any],
        employee_comparison: Dict[str, Any],
        jd_analysis: Dict[str, Any],
        additional_info: Dict[str, Any] = None
    ) -> str:
        """
        Format the complete candidate profile review.
        
        Args:
            candidate_data: Parsed candidate resume data
            employee_comparison: Results from employee comparison
            jd_analysis: Results from JD analysis
            additional_info: Additional candidate info (CTC, notice period, etc.)
        
        Returns:
            Formatted string output
        """
        additional_info = additional_info or {}
        
        output_lines = []
        
        # Comparison with Employees
        output_lines.append("Comparison with Our Employees")
        output_lines.append("")
        
        location = employee_comparison.get("location", "Not mentioned")
        output_lines.append(f"Location: {location}")
        output_lines.append("")
        
        edu_institute, edu_matches, edu_note = employee_comparison.get("education_institute", ("Not mentioned", "No", ""))
        output_lines.append(f"Education Institute: {edu_institute}")
        output_lines.append(f"Anyone from same Institution: {edu_matches}")
        if edu_note:
            output_lines.append(f"Note: {edu_note}")
        output_lines.append("")
        
        edu_spec, spec_matches = employee_comparison.get("education_specialization", ("Not mentioned", "No"))
        output_lines.append(f"Education Specialization: {edu_spec}")
        if spec_matches and spec_matches != "No":
            # Format: "Employee Name (Specialization), Employee Name (Specialization)"
            output_lines.append(f"Related Specialization: {spec_matches}")
        else:
            output_lines.append(f"Related Specialization: No")
        output_lines.append("")
        
        exp_companies, exp_matches = employee_comparison.get("experience_companies", ("Not mentioned", "No"))
        output_lines.append(f"Candidate Previous Experience: {exp_companies}")
        output_lines.append(f"Anyone from there: {exp_matches}")
        output_lines.append("")
        
        clients, common_clients = employee_comparison.get("clients", ("Not mentioned", "No"))
        output_lines.append(f"Previous Clients Served: {clients}")
        output_lines.append(f"Common Clients: {common_clients}")
        output_lines.append("")
        
        common_skills, unique_skills = employee_comparison.get("skills", ([], []))
        output_lines.append("Common Skills: " + (", ".join(common_skills) if common_skills and common_skills != ["No"] else "No"))
        output_lines.append("Unique Skills: " + (", ".join(unique_skills) if unique_skills else "No"))
        output_lines.append("")
        
        common_hobbies, unique_hobbies = employee_comparison.get("hobbies", ([], []))
        output_lines.append("Common Hobbies/Activities: " + (", ".join(common_hobbies) if common_hobbies and common_hobbies != ["No"] else "No"))
        output_lines.append("Unique Hobbies/Activities: " + (", ".join(unique_hobbies) if unique_hobbies else "No"))
        output_lines.append("")
        
        common_achievements, unique_achievements = employee_comparison.get("achievements", ([], []))
        output_lines.append("Common Achievements: " + (", ".join(common_achievements) if common_achievements and common_achievements != ["No"] else "No"))
        output_lines.append("Unique Achievements: " + (", ".join(unique_achievements) if unique_achievements else "No"))
        output_lines.append("")
        
        common_certs, unique_certs = employee_comparison.get("certifications", ([], []))
        output_lines.append("Common Certifications: " + (", ".join(common_certs) if common_certs and common_certs != ["No"] else "No"))
        output_lines.append("Unique Certifications: " + (", ".join(unique_certs) if unique_certs else "No"))
        output_lines.append("")
        
        common_langs, unique_langs = employee_comparison.get("languages", ([], []))
        output_lines.append("Common Languages: " + (", ".join(common_langs) if common_langs and common_langs != ["Not mentioned"] else "Not mentioned"))
        output_lines.append("Unique Languages: " + (", ".join(unique_langs) if unique_langs else "No"))
        output_lines.append("")
        
        cultural_fit_score = employee_comparison.get("cultural_fit_score", 0)
        output_lines.append(f"Cultural Fit Score: {cultural_fit_score}")
        output_lines.append("")
        
        # JD Analysis
        output_lines.append("Job Description Analysis")
        output_lines.append("")
        jd_score = jd_analysis.get("jd_relevance_score", 0)
        jd_suggestion = jd_analysis.get("ai_suggestion", "")
        output_lines.append(f"JD Relevance Score: {jd_score}")
        output_lines.append(f"AI Suggestion: {jd_suggestion}")
        
        return "\n".join(output_lines)

