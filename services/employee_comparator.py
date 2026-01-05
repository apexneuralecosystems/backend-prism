"""Employee Comparator Module - Compares candidate data with existing employee data."""
from typing import Dict, List, Any, Set
import re


class EmployeeComparator:
    """Compares candidate profile with existing employees."""
    
    def __init__(self, employees_data: List[Dict[str, Any]]):
        """
        Initialize with employee data.
        
        Args:
            employees_data: List of employee dictionaries (parsed resume format)
        """
        self.employees = employees_data

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, remove extra spaces)."""
        if not text or text == "info not available in resume":
            return ""
        return re.sub(r'\s+', ' ', text.lower().strip())

    def extract_school_location(self, candidate_data: Dict[str, Any]) -> str:
        """
        Extract location from school information (10th, 12th, diploma).
        Returns format: "City, State"
        """
        schools = candidate_data.get("School", [])
        if isinstance(schools, str):
            schools = [schools] if schools != "info not available in resume" else []
        
        # Try to extract location from school names or use State/City
        state = candidate_data.get("State", "")
        city = candidate_data.get("City", "")
        
        if state and city:
            return f"{city}, {state}"
        elif state:
            return state
        elif city:
            return city
        return "Not mentioned"

    def compare_education_institute(self, candidate_data: Dict[str, Any]) -> tuple:
        """
        Compare education institutes.
        Returns: (candidate_institute, matching_employees, note)
        """
        candidate_edu = candidate_data.get("Education", [])
        if not candidate_edu:
            return ("Not mentioned", [], "")

        candidate_colleges = []
        for edu in candidate_edu:
            college = edu.get("College", "")
            if college and college != "info not available in resume":
                candidate_colleges.append(self.normalize_text(college))

        if not candidate_colleges:
            return ("Not mentioned", [], "")

        # Get primary institute (first one) with location
        primary_edu = candidate_data["Education"][0]
        primary_college = primary_edu.get("College", "Not mentioned")

        # Add location if available
        state = candidate_data.get("State", "")
        city = candidate_data.get("City", "")
        if primary_college != "Not mentioned":
            if city and state:
                # Check if location is already in college name
                if city.lower() not in primary_college.lower() and state.lower() not in primary_college.lower():
                    primary_college = f"{primary_college} {city}, {state}"
            elif state:
                if state.lower() not in primary_college.lower():
                    primary_college = f"{primary_college} {state}"

        matching_employees = []
        same_university_different_branch = []

        for emp in self.employees:
            emp_name = emp.get("Name", "")
            emp_edu = emp.get("Education", [])
            
            for edu in emp_edu:
                emp_college = self.normalize_text(edu.get("College", ""))
                
                # Check for exact match
                for cand_college in candidate_colleges:
                    if emp_college and cand_college:
                        # Exact match
                        if emp_college == cand_college:
                            if emp_name not in matching_employees:
                                matching_employees.append(emp_name)
                        # Same university, different branch
                        elif self._is_same_university(emp_college, cand_college):
                            if emp_name not in same_university_different_branch:
                                same_university_different_branch.append(emp_name)

        note = ""
        if same_university_different_branch:
            note = f"Same university, different branch: {', '.join(same_university_different_branch)}"

        if matching_employees:
            result = ", ".join(matching_employees)
            if note:
                result += f" ({note})"
            return (primary_college, result, note)

        if same_university_different_branch:
            return (primary_college, "No", note)

        return (primary_college, "No", "")

    def _is_same_university(self, college1: str, college2: str) -> bool:
        """Check if two colleges are the same university but different branches."""
        # Extract university name (remove location/branch info)
        uni1 = re.sub(r',\s*\w+.*$', '', college1).strip()
        uni2 = re.sub(r',\s*\w+.*$', '', college2).strip()
        
        # Check for common university patterns
        common_unis = ["vit", "vellore institute", "iit", "nit", "bits"]
        for uni in common_unis:
            if uni in uni1 and uni in uni2:
                return True
        
        return False

    def compare_education_specialization(self, candidate_data: Dict[str, Any]) -> tuple:
        """
        Compare education specializations.
        Returns: (candidate_spec, matching_employees)
        """
        candidate_edu = candidate_data.get("Education", [])
        if not candidate_edu:
            return ("Not mentioned", "No")

        candidate_specs = []
        for edu in candidate_edu:
            spec = edu.get("Specialization", "")
            if spec and spec != "info not available in resume":
                candidate_specs.append(self.normalize_text(spec))

        if not candidate_specs:
            return ("Not mentioned", "No")

        primary_spec = candidate_data["Education"][0].get("Specialization", "Not mentioned")

        matching_employees = []
        matched_names = set()

        for emp in self.employees:
            emp_name = emp.get("Name", "")
            emp_edu = emp.get("Education", [])
            
            for edu in emp_edu:
                emp_spec = edu.get("Specialization", "")
                emp_spec_normalized = self.normalize_text(emp_spec)
                
                for cand_spec in candidate_specs:
                    if emp_spec_normalized and cand_spec:
                        # Exact match
                        if emp_spec_normalized == cand_spec:
                            if emp_name not in matched_names:
                                matched_names.add(emp_name)
                                matching_employees.append(f"{emp_name} ({emp_spec})")
                        # Related specialization (AI, ML, CS related)
                        elif self._is_related_specialization(emp_spec_normalized, cand_spec):
                            if emp_name not in matched_names:
                                matched_names.add(emp_name)
                                matching_employees.append(f"{emp_name} ({emp_spec})")

        if matching_employees:
            result = ", ".join(matching_employees)
            # Add note about related specializations in AI/ML fields
            primary_spec_lower = primary_spec.lower()
            if any(keyword in primary_spec_lower for keyword in ["ai", "ml", "machine learning", "artificial intelligence"]):
                result += ", closely related specializations in AI and ML fields"
            return (primary_spec, result)

        return (primary_spec, "No")

    def _is_related_specialization(self, spec1: str, spec2: str) -> bool:
        """Check if specializations are related (AI/ML/CS related)."""
        ai_ml_keywords = ["ai", "artificial intelligence", "ml", "machine learning",
                          "deep learning", "data science", "computer science", "cse"]
        
        spec1_has_keywords = any(keyword in spec1 for keyword in ai_ml_keywords)
        spec2_has_keywords = any(keyword in spec2 for keyword in ai_ml_keywords)
        
        return spec1_has_keywords and spec2_has_keywords

    def compare_experience_companies(self, candidate_data: Dict[str, Any]) -> tuple:
        """
        Compare experience companies (exact match only).
        Returns: (candidate_companies, matching_employees)
        """
        candidate_exp = candidate_data.get("Experience", [])
        if not candidate_exp or candidate_exp == "info not available in resume":
            return ("Not mentioned", "No")

        candidate_companies = []
        for exp in candidate_exp:
            company = exp.get("Company", "")
            if company and company != "info not available in resume":
                candidate_companies.append(self.normalize_text(company))

        if not candidate_companies:
            return ("Not mentioned", "No")

        company_list = [exp.get("Company", "") for exp in candidate_exp if exp.get("Company", "") != "info not available in resume"]

        matching_employees = []

        for emp in self.employees:
            emp_name = emp.get("Name", "")
            emp_exp = emp.get("Experience", [])
            
            if isinstance(emp_exp, str) and emp_exp == "info not available in resume":
                continue
            
            for exp in emp_exp:
                emp_company = self.normalize_text(exp.get("Company", ""))
                
                for cand_company in candidate_companies:
                    if emp_company and cand_company and emp_company == cand_company:
                        if emp_name not in matching_employees:
                            matching_employees.append(emp_name)

        company_str = ", ".join(company_list) if company_list else "Not mentioned"

        if matching_employees:
            return (company_str, ", ".join(matching_employees))

        return (company_str, "No")

    def extract_clients(self, candidate_data: Dict[str, Any]) -> tuple:
        """
        Extract clients from experience descriptions.
        Returns: (candidate_clients, common_clients)
        """
        candidate_exp = candidate_data.get("Experience", [])
        if not candidate_exp or candidate_exp == "info not available in resume":
            return ("Not mentioned", "No")

        candidate_clients = []
        for exp in candidate_exp:
            desc = exp.get("Description", "")
            if desc and desc != "info not available in resume":
                # Extract client names (look for "Client:" pattern)
                client_match = re.search(r'[Cc]lient[:\s]+([A-Z][A-Za-z\s&.,]+)', desc)
                if client_match:
                    candidate_clients.append(client_match.group(1).strip())

        if not candidate_clients:
            return ("Not mentioned", "No")

        # Compare with employees
        common_clients = []
        for emp in self.employees:
            emp_exp = emp.get("Experience", [])
            if isinstance(emp_exp, str):
                continue
            
            for exp in emp_exp:
                desc = exp.get("Description", "")
                if desc:
                    for client in candidate_clients:
                        if self.normalize_text(client) in self.normalize_text(desc):
                            if client not in common_clients:
                                common_clients.append(client)

        clients_str = ", ".join(candidate_clients)

        if common_clients:
            return (clients_str, ", ".join(common_clients))

        return (clients_str, "No")

    def compare_skills(self, candidate_data: Dict[str, Any]) -> tuple:
        """
        Compare skills (from Skills section only).
        Returns: (common_skills, unique_skills)
        """
        candidate_skills = candidate_data.get("Skills", [])
        if isinstance(candidate_skills, str):
            candidate_skills = []

        candidate_skills_normalized = {self.normalize_text(skill) for skill in candidate_skills}

        all_employee_skills = set()
        for emp in self.employees:
            emp_skills = emp.get("Skills", [])
            if isinstance(emp_skills, str):
                continue
            for skill in emp_skills:
                all_employee_skills.add(self.normalize_text(skill))

        common_skills = []
        unique_skills = []

        for skill in candidate_skills:
            skill_norm = self.normalize_text(skill)
            if skill_norm in all_employee_skills:
                if skill not in common_skills:
                    common_skills.append(skill)
            else:
                # Check if it's truly unique (not closely related)
                if not self._is_similar_skill(skill_norm, all_employee_skills):
                    unique_skills.append(skill)

        return (common_skills if common_skills else ["No"], unique_skills if unique_skills else [])

    def _is_similar_skill(self, skill: str, skill_set: Set[str]) -> bool:
        """Check if skill is similar to any in the set."""
        # Normalize variations
        skill_lower = skill.lower()
        
        # Common skill variations
        variations = {
            "python": ["python3", "python 3"],
            "javascript": ["js", "node.js", "nodejs"],
            "react": ["reactjs", "react.js"],
            "mysql": ["sql", "plsql", "pl-sql"],
            "mongodb": ["mongo", "nosql"],
            "aws": ["amazon web services", "ec2", "s3"],
            "machine learning": ["ml", "scikit-learn", "sklearn"],
            "deep learning": ["dl", "tensorflow", "pytorch", "keras"],
        }
        
        for base, variants in variations.items():
            if base in skill_lower:
                for variant in variants:
                    if any(variant in s for s in skill_set):
                        return True
            if any(variant in skill_lower for variant in variants):
                if any(base in s for s in skill_set):
                    return True
        
        return False

    def compare_hobbies(self, candidate_data: Dict[str, Any]) -> tuple:
        """Compare hobbies/activities."""
        candidate_hobbies = candidate_data.get("Activities_Hobbies", [])
        if isinstance(candidate_hobbies, str) or candidate_hobbies == "info not available in resume":
            return (["No"], [])

        candidate_hobbies_normalized = {self.normalize_text(h) for h in candidate_hobbies}

        all_employee_hobbies = set()
        for emp in self.employees:
            emp_hobbies = emp.get("Activities_Hobbies", [])
            if isinstance(emp_hobbies, str):
                continue
            if isinstance(emp_hobbies, list):
                for hobby in emp_hobbies:
                    all_employee_hobbies.add(self.normalize_text(hobby))

        common = []
        unique = []

        for hobby in candidate_hobbies:
            hobby_norm = self.normalize_text(hobby)
            if hobby_norm in all_employee_hobbies:
                if hobby not in common:
                    common.append(hobby)
            else:
                unique.append(hobby)

        return (common if common else ["No"], unique if unique else [])

    def compare_achievements(self, candidate_data: Dict[str, Any]) -> tuple:
        """Compare achievements (external/individual only, exclude project/work)."""
        candidate_achievements = candidate_data.get("Achievements", [])
        if isinstance(candidate_achievements, str) or candidate_achievements == "info not available in resume":
            return (["No"], [])

        # Filter out project/work achievements
        external_achievements = []
        for ach in candidate_achievements:
            ach_lower = ach.lower()
            # Exclude project-related achievements
            if not any(keyword in ach_lower for keyword in ["project", "developed", "built", "created", "implemented"]):
                external_achievements.append(ach)

        if not external_achievements:
            return (["No"], [])

        candidate_ach_normalized = {self.normalize_text(a) for a in external_achievements}

        all_employee_achievements = set()
        for emp in self.employees:
            emp_ach = emp.get("Achievements", [])
            if isinstance(emp_ach, str):
                continue
            if isinstance(emp_ach, list):
                for ach in emp_ach:
                    ach_lower = ach.lower()
                    if not any(keyword in ach_lower for keyword in ["project", "developed", "built", "created", "implemented"]):
                        all_employee_achievements.add(self.normalize_text(ach))

        common = []
        unique = []

        for ach in external_achievements:
            ach_norm = self.normalize_text(ach)
            if ach_norm in all_employee_achievements:
                if ach not in common:
                    common.append(ach)
            else:
                unique.append(ach)

        return (common if common else ["No"], unique if unique else [])

    def compare_certifications(self, candidate_data: Dict[str, Any]) -> tuple:
        """Compare certifications."""
        candidate_certs = candidate_data.get("Certifications", [])
        if isinstance(candidate_certs, str) or candidate_certs == "info not available in resume":
            return (["No"], [])

        candidate_certs_normalized = {self.normalize_text(c) for c in candidate_certs}

        all_employee_certs = set()
        for emp in self.employees:
            emp_certs = emp.get("Certifications", [])
            if isinstance(emp_certs, str):
                continue
            if isinstance(emp_certs, list):
                for cert in emp_certs:
                    all_employee_certs.add(self.normalize_text(cert))

        common = []
        unique = []

        for cert in candidate_certs:
            cert_norm = self.normalize_text(cert)
            if cert_norm in all_employee_certs:
                if cert not in common:
                    common.append(cert)
            else:
                unique.append(cert)

        return (common if common else ["No"], unique if unique else [])

    def compare_languages(self, candidate_data: Dict[str, Any]) -> tuple:
        """Compare languages (explicitly mentioned only)."""
        candidate_langs = candidate_data.get("Languages", [])
        if isinstance(candidate_langs, str) or candidate_langs == "info not available in resume":
            return (["Not mentioned"], [])

        # Filter out "info not available in resume" entries
        candidate_langs = [lang for lang in candidate_langs if lang != "info not available in resume"]

        if not candidate_langs:
            return (["Not mentioned"], [])

        candidate_langs_normalized = {self.normalize_text(lang) for lang in candidate_langs}

        all_employee_langs = set()
        for emp in self.employees:
            emp_langs = emp.get("Languages", [])
            if isinstance(emp_langs, str):
                continue
            if isinstance(emp_langs, list):
                for lang in emp_langs:
                    if lang != "info not available in resume":
                        all_employee_langs.add(self.normalize_text(lang))

        common = []
        unique = []

        for lang in candidate_langs:
            lang_norm = self.normalize_text(lang)
            if lang_norm in all_employee_langs:
                if lang not in common:
                    common.append(lang)
            else:
                unique.append(lang)

        return (common if common else ["Not mentioned"], unique if unique else [])

    def calculate_cultural_fit_score(self, candidate_data: Dict[str, Any]) -> int:
        """
        Calculate cultural fit score (0-100) based on various factors.
        """
        score = 0
        max_score = 100

        # Education institute match (20 points)
        _, emp_matches, _ = self.compare_education_institute(candidate_data)
        if emp_matches and emp_matches != "No":
            score += 20
        elif emp_matches == "No" and "different branch" in str(emp_matches):
            score += 10

        # Education specialization match (20 points)
        _, spec_matches = self.compare_education_specialization(candidate_data)
        if spec_matches and spec_matches != "No":
            score += 20

        # Skills overlap (30 points)
        common_skills, _ = self.compare_skills(candidate_data)
        if common_skills and common_skills != ["No"]:
            skill_overlap = len(common_skills) / max(len(candidate_data.get("Skills", [])), 1)
            score += min(30, int(skill_overlap * 30))

        # Experience company match (10 points)
        _, exp_matches = self.compare_experience_companies(candidate_data)
        if exp_matches and exp_matches != "No":
            score += 10

        # Languages common (10 points)
        common_langs, _ = self.compare_languages(candidate_data)
        if common_langs and common_langs != ["Not mentioned"]:
            score += min(10, len(common_langs) * 3)

        # Certifications common (10 points)
        common_certs, _ = self.compare_certifications(candidate_data)
        if common_certs and common_certs != ["No"]:
            score += min(10, len(common_certs) * 2)

        return min(score, max_score)

