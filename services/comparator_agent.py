"""Main Comparator Agent - Orchestrates employee comparison and JD analysis."""
from typing import Dict, Any, List
from .employee_comparator import EmployeeComparator
from .jd_analyzer import JDAnalyzer
from .output_formatter import OutputFormatter


class ComparatorAgent:
    """Main agent that compares candidate with employees and analyzes JD fit."""
    
    def __init__(self, employees_data: List[Dict[str, Any]], model_name: str = "gpt-4o-mini"):
        """
        Initialize the comparator agent.
        
        Args:
            employees_data: List of employee dictionaries (parsed resume format)
            model_name: OpenAI model name to use
        """
        self.employee_comparator = EmployeeComparator(employees_data)
        self.jd_analyzer = JDAnalyzer(model_name=model_name)
        self.output_formatter = OutputFormatter()

    def process(
        self,
        candidate_data: Dict[str, Any],
        job_description: str,
        additional_info: Dict[str, Any] = None
    ) -> str:
        """
        Process candidate data and generate comparison report.
        
        Args:
            candidate_data: Already parsed candidate resume data (dictionary)
            job_description: Job description text
            additional_info: Additional candidate information (CTC, notice period, role, etc.)
        
        Returns:
            Formatted comparison report string
        """
        additional_info = additional_info or {}
        
        # Step 1: Compare with employees
        print("Comparing with employees...")
        employee_comparison = self._compare_with_employees(candidate_data)
        
        # Step 2: Analyze JD fit
        print("Analyzing JD fit...")
        jd_analysis = self.jd_analyzer.analyze(candidate_data, job_description)
        
        # Step 3: Format output
        print("Formatting output...")
        output = self.output_formatter.format(
            candidate_data,
            employee_comparison,
            jd_analysis,
            additional_info
        )
        
        return output

    def _compare_with_employees(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform all employee comparisons."""
        location = self.employee_comparator.extract_school_location(candidate_data)
        edu_institute, edu_matches, edu_note = self.employee_comparator.compare_education_institute(candidate_data)
        edu_spec, spec_matches = self.employee_comparator.compare_education_specialization(candidate_data)
        exp_companies, exp_matches = self.employee_comparator.compare_experience_companies(candidate_data)
        clients, common_clients = self.employee_comparator.extract_clients(candidate_data)
        common_skills, unique_skills = self.employee_comparator.compare_skills(candidate_data)
        common_hobbies, unique_hobbies = self.employee_comparator.compare_hobbies(candidate_data)
        common_achievements, unique_achievements = self.employee_comparator.compare_achievements(candidate_data)
        common_certs, unique_certs = self.employee_comparator.compare_certifications(candidate_data)
        common_langs, unique_langs = self.employee_comparator.compare_languages(candidate_data)
        cultural_fit_score = self.employee_comparator.calculate_cultural_fit_score(candidate_data)
        
        return {
            "location": location,
            "education_institute": (edu_institute, edu_matches, edu_note),
            "education_specialization": (edu_spec, spec_matches),
            "experience_companies": (exp_companies, exp_matches),
            "clients": (clients, common_clients),
            "skills": (common_skills, unique_skills),
            "hobbies": (common_hobbies, unique_hobbies),
            "achievements": (common_achievements, unique_achievements),
            "certifications": (common_certs, unique_certs),
            "languages": (common_langs, unique_langs),
            "cultural_fit_score": cultural_fit_score
        }

