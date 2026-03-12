# Offer letter template

Place **Offer_Letter_Template.docx** in this folder for auto-generated offer letters.

Alternatively:
- Set `OFFER_LETTER_TEMPLATE_PATH` in `.env` to an absolute path or path relative to the backend directory.
- Or place the file at the **project root**: `Offer_Letter_Template.docx` (same folder as `backend/` and `frontend/`).

## Placeholders (use exactly in the DOCX)

- **Company:** `[COMPANY NAME]`, `[Company Name]`, `[Company Address, City, State – ZIP]`, `[Office Address / Remote / Hybrid]`, `[Phone]`, `[HR Phone Number]`, `[www.company.com]`, `[HR Email]`
- **Job:** `Ref: job id` → becomes `Ref: <job_id>`, `[Job Title]`, `[Full-Time / Part-Time / Contract]`
- **Candidate:** `[Candidate Full Name]`, `[Candidate First Name]`, `[candidate@email.com]`
- **Dates / offer:** `[DD Month YYYY]` (end date or `"Not mentioned"` if empty), `[e.g., 12 months / Permanent / Probation Period: 3 months]`, `[e.g., ₹X,XX,XXX per annum / $XX,XXX per year]`, `[Monthly / Bi-weekly / Weekly]`, `[Health Insurance, PF, Gratuity, ESOPs, etc.]`

Example footer in the DOCX:

Human Resources Manager  
[Company Name]  
Email: [HR Email] | Phone: [HR Phone Number]

Curly style is also supported: `{{COMPANY_NAME}}`, `{{CANDIDATE_NAME}}`, `{{ROLE}}`, `{{END_DATE}}`, `{{EMPLOYMENT_TYPE}}`, etc.

Organization logo: set `logo_path` in organization data to an S3 URL or local path; it will be inserted in the header.
