import smtplib
import os
from pathlib import Path
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from string import Template
from dotenv import load_dotenv

# Load environment variables - explicitly set path to backend/.env
backend_dir = Path(__file__).parent.parent
env_path = backend_dir / ".env"
print(f"🔍 [email_service] Loading .env from: {env_path}")
print(f"🔍 [email_service] .env file exists: {env_path.exists()}")

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print("✅ [email_service] .env file loaded successfully")
else:
    print(f"⚠️  [email_service] .env file not found at {env_path}, trying current directory")
    load_dotenv(override=True)
    
# Debug: Show all loaded email-related env vars (TO_EMAIL comes from form, not .env)
print(f"🔍 [email_service] Environment variables after loading:")
for key in ["FROM_EMAIL", "EMAIL_PASSWORD", "SMTP_SERVER", "SMTP_PORT"]:
    value = os.getenv(key)
    if value:
        if "PASSWORD" in key:
            masked = "***" + value[-4:] if len(value) > 4 else "***"
            print(f"   {key} = {masked}")
        else:
            print(f"   {key} = {value[:30]}..." if len(value) > 30 else f"   {key} = {value}")
    else:
        print(f"   {key} = ❌ NOT FOUND")
print(f"   Note: TO_EMAIL will come from form input (requester's email)")


def send_mail(
    to_emails,
    subject,
    message,
    password,
    from_email,
    html_content=None,
    smtp_server=None,
    smtp_port=None
):
    """
    Send email using SMTP
    
    Args:
        to_emails: Email address(es) to send to (str or list)
        subject: Email subject
        message: Plain text message
        password: SMTP password
        from_email: Sender email address
        html_content: Optional HTML content
        smtp_server: SMTP server address (defaults to env var or smtp.gmail.com)
        smtp_port: SMTP server port (defaults to env var or 587)
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    # Get SMTP settings from environment if not provided
    if smtp_server is None:
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    if smtp_port is None:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    
    # Ensure to_emails is a list
    if isinstance(to_emails, str):
        to_emails = [to_emails]

    try:
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = ', '.join(to_emails)  # Use the to_emails parameter (user's email for OTP, fixed email for demo requests)

        # Add plain text version
        text_part = MIMEText(message, 'plain')
        msg.attach(text_part)

        # Add HTML version if provided
        if html_content:
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

        # Connect to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()

        print('✅ Successfully sent the email.')
        return True
    except smtplib.SMTPException as e:
        print(f"❌ Error sending email: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error sending email: {e}")
        return False


def get_email_template():
    """
    Get the HTML email template for demo requests
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Demo Request</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #111827;
            margin: 0;
            padding: 40px 20px;
        }
        .container {
            max-width: 680px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            padding: 40px 32px;
            text-align: center;
            position: relative;
        }
        .header-icon {
            width: 70px;
            height: 70px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 36px;
            backdrop-filter: blur(10px);
        }
        .header h1 {
            margin: 0 0 8px 0;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        .header p {
            margin: 0;
            font-size: 15px;
            opacity: 0.95;
            font-weight: 500;
        }
        .content {
            padding: 36px 32px;
        }
        .content p {
            margin: 0 0 16px;
            font-size: 15px;
            line-height: 1.7;
            color: #374151;
        }
        .content p.greeting {
            font-size: 16px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 20px;
        }
        .details-box {
            background: linear-gradient(to bottom right, #f8fafc, #f1f5f9);
            border: 2px solid #e5e7eb;
            border-radius: 16px;
            margin: 28px 0;
            overflow: hidden;
        }
        .details-header {
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            padding: 14px 20px;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 0.3px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        tr {
            border-bottom: 1px solid #e5e7eb;
            transition: background 0.2s;
        }
        tr:last-child {
            border-bottom: none;
        }
        tr:hover {
            background-color: #f9fafb;
        }
        td {
            padding: 16px 20px;
            vertical-align: top;
            font-size: 14px;
        }
        td.label {
            font-weight: 700;
            color: #1f2937;
            width: 200px;
            position: relative;
        }
        td.label:before {
            content: '•';
            color: #667eea;
            font-weight: bold;
            display: inline-block;
            width: 1em;
            margin-left: -1em;
            position: absolute;
            left: 8px;
        }
        td.value {
            color: #4b5563;
            font-weight: 500;
        }
        a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s;
        }
        a:hover {
            color: #764ba2;
            text-decoration: underline;
        }
        .action-note {
            background: linear-gradient(to right, #dbeafe, #e0e7ff);
            border-left: 4px solid #3b82f6;
            padding: 16px 20px;
            border-radius: 8px;
            margin: 24px 0;
        }
        .action-note p {
            margin: 0;
            color: #1e40af;
            font-weight: 600;
            font-size: 14px;
        }
        .signature {
            margin-top: 32px;
            padding-top: 20px;
            border-top: 2px solid #e5e7eb;
        }
        .signature p {
            margin: 4px 0;
            color: #1f2937;
        }
        .footer {
            padding: 24px 32px;
            border-top: 2px solid #e5e7eb;
            font-size: 13px;
            color: #6b7280;
            text-align: center;
            background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
        }
        .footer a {
            color: #667eea;
            font-weight: 600;
        }
        strong {
            font-weight: 700;
            color: #111827;
        }
        .emoji {
            font-size: 20px;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-icon">🎯</div>
            <h1>PRISM Demo Request</h1>
            <p>New Platform Demo Request Received</p>
        </div>
        <div class="content">
            <p class="greeting">👋 Hi Team,</p>
            <p>We've received a new demo request for the <strong>Prism Platform</strong>. Please review the requester details below and prepare the demo accordingly.</p>
            
            <div class="details-box">
                <div class="details-header">📋 Requester Information</div>
                <table>
                    <tr>
                        <td class="label">Name</td>
                        <td class="value">${name}</td>
                    </tr>
                    <tr>
                        <td class="label">Email</td>
                        <td class="value"><a href="mailto:${email}">${email}</a></td>
                    </tr>
                    <tr>
                        <td class="label">Phone</td>
                        <td class="value">${phone}</td>
                    </tr>
                    <tr>
                        <td class="label">Company / Organization</td>
                        <td class="value">${companyName}</td>
                    </tr>
                    <tr>
                        <td class="label">Position</td>
                        <td class="value">${position}</td>
                    </tr>
                    <tr>
                        <td class="label">Preferred Date & Time</td>
                        <td class="value">${date}, ${time}</td>
                    </tr>
                    <tr>
                        <td class="label">Additional Notes</td>
                        <td class="value">${comments}</td>
                    </tr>
                </table>
            </div>
            
            <div class="action-note">
                <p>⚡ Action Required: Please evaluate the request and prepare the demo. Confirm scheduling in reply to this thread.</p>
            </div>
            
            <div class="signature">
                <p><strong>Best regards,</strong></p>
                <p><strong style="color: #667eea; font-size: 16px;">PRISM</strong></p>
                <p style="color: #6b7280; font-size: 13px;">APEXNEURAL</p>
            </div>
        </div>
        
        <div class="footer">
            <p style="margin: 0 0 8px 0;">This message was sent to the PRISM team.</p>
            <p style="margin: 0;">Need help? Contact <a href="mailto:support@apexneural.cloud">support@apexneural.cloud</a></p>
        </div>
    </div>
</body>
</html>"""


def send_demo_request_email(
    name: str,
    email: str,
    phone: str,
    company_name: str,
    position: str,
    date: str,
    time: str,
    comments: str = ""
):
    """
    Send demo request email notification
    
    Args:
        name: Requester's name
        email: Requester's email
        phone: Requester's phone number
        company_name: Company/Organization name
        position: Requester's position
        date: Preferred date
        time: Preferred time
        comments: Additional comments/notes
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    # Get email credentials from environment variables (FROM_EMAIL and EMAIL_PASSWORD only)
    # TO_EMAIL comes from the form input (requester's email)
    print(f"\n🔍 [send_demo_request_email] Starting email send process...")
    print(f"🔍 [send_demo_request_email] Current working directory: {os.getcwd()}")
    
    # Try both standard uppercase and the user's mixed case variant
    from_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
    password = os.getenv("EMAIL_PASSWORD")
    # Demo request emails go to fixed recipient (akshaay.kg@apexneural.com)
    # The requester's email is included in the email body, not as recipient
    to_email = os.getenv("TO_EMAIL", "akshaay.kg@apexneural.com")  # Fixed recipient for demo requests
    
    # Detailed debug logging
    print(f"\n🔍 [send_demo_request_email] Checking credentials:")
    print(f"   FROM_EMAIL (from .env): {repr(from_email)}")
    print(f"   EMAIL_PASSWORD (from .env): {'✅ Found (length: ' + str(len(password)) + ')' if password else '❌ MISSING (None or empty)'}")
    print(f"   TO_EMAIL (fixed recipient for demo requests): {repr(to_email)}")
    print(f"   Requester's email (included in body): {repr(email)}")
    
    # Show all environment variables
    print(f"\n🔍 [send_demo_request_email] All environment variables:")
    print(f"   Total env vars: {len(os.environ)}")
    email_related = {k: v for k, v in os.environ.items() if 'EMAIL' in k.upper() or 'FROM' in k.upper() or 'SMTP' in k.upper()}
    if email_related:
        print(f"   Found {len(email_related)} email-related variables:")
        for key, value in sorted(email_related.items()):
            if "PASSWORD" in key:
                masked = "***" + value[-4:] if len(value) > 4 else "***"
                print(f"      {key} = {masked} (length: {len(value)})")
            else:
                print(f"      {key} = {value}")
    else:
        print("   ❌ No email-related environment variables found!")
        print("   Showing first 10 env vars for debugging:")
        for i, (key, value) in enumerate(list(os.environ.items())[:10]):
            print(f"      {key} = {value[:50]}...")
    
    if not from_email or not password:
        print(f"\n❌ [send_demo_request_email] Email credentials from .env not found!")
        print(f"   FROM_EMAIL is missing: {not from_email} (value: {repr(from_email)})")
        print(f"   EMAIL_PASSWORD is missing: {not password} (value: {repr(password)})")
        return False
    
    if not to_email:
        print(f"\n❌ [send_demo_request_email] TO_EMAIL (recipient email) is missing!")
        return False
    
    print(f"✅ [send_demo_request_email] All credentials found, proceeding to send email...")
    
    # Get email template
    html_template = get_email_template()
    
    # Replace placeholders in template using Template to avoid CSS brace conflicts
    html_content = Template(html_template).substitute(
        name=name,
        email=email,
        phone=phone,
        companyName=company_name,
        position=position,
        date=date,
        time=time,
        comments=comments if comments else "N/A"
    )
    
    # Create plain text version
    plain_text = f"""
PRISM — Demo Request

Hi Team,

We've received a new demo request for the Prism Platform. Please review the requester details below and prepare the demo accordingly.

Name: {name}
Email: {email}
Phone: {phone}
Company / Organization: {company_name}
Position: {position}
Preferred Date & Time: {date}, {time}
Additional Notes: {comments if comments else "N/A"}

Kindly evaluate the request and prepare the demo. Please confirm scheduling in reply to this thread and advise if any additional information is required.

Best regards,
PRISM
APEXNEURAL
"""
    
    # Send email
    subject = "Request for Prism Platform Demo"
    
    return send_mail(
        to_emails=to_email,
        subject=subject,
        message=plain_text,
        password=password,
        from_email=from_email,
        html_content=html_content
    )



async def send_otp_email(email: str, otp: str, purpose: str = "signup"):
    """
    Send OTP email for authentication
    
    Args:
        email: Recipient email address
        otp: 6-digit OTP code
        purpose: Either "signup" or "reset" for password reset
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    print(f"\n🔍 [send_otp_email] Starting OTP email send...")
    print(f"🔍 [send_otp_email] Recipient email (from form): {repr(email)}")
    print(f"🔍 [send_otp_email] Purpose: {purpose}")
    
    # Get credentials from environment
    from_email = os.getenv("FROM_EMAIL")
    password = os.getenv("EMAIL_PASSWORD")
    
    print(f"🔍 [send_otp_email] FROM_EMAIL: {from_email}")
    print(f"🔍 [send_otp_email] PASSWORD: {'SET' if password else 'NOT SET'}")
    
    if not from_email or not password:
        print(f"❌ [send_otp_email] Email credentials not found!")
        return False
    
    # Verify recipient email is provided (from user's form input)
    if not email:
        print(f"❌ [send_otp_email] Recipient email is missing!")
        return False
    
    print(f"✅ [send_otp_email] Sending OTP to user's email: {email}")
    
    subject = "Verify Your Email - PRISM" if purpose == "signup" else "Reset Your Password - PRISM"
    
    purpose_text = "verify your email" if purpose == "signup" else "reset your password"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM OTP Verification</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 40px 20px;
            color: #111827;
        }}
        .container {{
            max-width: 640px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            padding: 40px 32px;
            text-align: center;
            position: relative;
        }}
        .header-icon {{
            width: 80px;
            height: 80px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 42px;
            backdrop-filter: blur(10px);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }}
        .header p {{
            margin: 0;
            font-size: 15px;
            opacity: 0.95;
            font-weight: 500;
        }}
        .content {{
            padding: 40px 36px;
        }}
        .content h2 {{
            margin: 0 0 24px 0;
            font-size: 24px;
            font-weight: 700;
            color: #111827;
            text-align: center;
        }}
        .content p {{
            margin: 0 0 18px 0;
            font-size: 15px;
            line-height: 1.7;
            color: #4b5563;
        }}
        .content p.greeting {{
            font-size: 16px;
            font-weight: 600;
            color: #111827;
        }}
        .otp-box {{
            background: linear-gradient(135deg, #f0f7ff 0%, #e0f2fe 100%);
            border: 3px solid #667eea;
            border-radius: 20px;
            padding: 40px 24px;
            text-align: center;
            margin: 32px 0;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
        }}
        .otp-label {{
            margin: 0 0 20px 0;
            color: #1f2937;
            font-size: 15px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .otp-code {{
            font-size: 48px;
            font-weight: 800;
            color: #667eea;
            letter-spacing: 16px;
            margin: 20px 0;
            font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
            text-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
        }}
        .otp-validity {{
            margin: 20px 0 0 0;
            color: #6b7280;
            font-size: 14px;
            font-weight: 600;
            background: white;
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
        }}
        .warning-box {{
            background: linear-gradient(to right, #fef2f2, #fee2e2);
            border-left: 4px solid #ef4444;
            padding: 18px 20px;
            margin: 28px 0;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
        }}
        .warning-box p {{
            margin: 0;
            color: #991b1b;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .info-note {{
            background: linear-gradient(to right, #f0fdf4, #dcfce7);
            border-left: 4px solid #22c55e;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 20px 0;
        }}
        .info-note p {{
            margin: 0;
            color: #166534;
            font-size: 14px;
            font-weight: 500;
        }}
        .footer {{
            background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
            padding: 28px 32px;
            text-align: center;
            font-size: 13px;
            color: #6b7280;
            border-top: 2px solid #e5e7eb;
        }}
        .footer p {{
            margin: 6px 0;
        }}
        .footer strong {{
            color: #667eea;
            font-size: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-icon">🔐</div>
            <h1>PRISM</h1>
            <p>Secure Authentication</p>
        </div>
        <div class="content">
            <h2>Your Verification Code</h2>
            <p class="greeting">👋 Hello,</p>
            <p>You requested to {purpose_text}. Please use the following verification code to complete your request securely:</p>
            
            <div class="otp-box">
                <p class="otp-label">🔑 Your OTP Code</p>
                <div class="otp-code">{otp}</div>
                <p class="otp-validity">⏰ Valid for 10 minutes only</p>
            </div>
            
            <div class="info-note">
                <p>✅ Enter this code in the verification form to proceed with your request.</p>
            </div>
            
            <div class="warning-box">
                <p><span style="font-size: 18px;">⚠️</span> <strong>Security Notice:</strong> Never share this code with anyone. PRISM staff will never ask for your OTP.</p>
            </div>
            
            <p style="color: #9ca3af; font-size: 13px; text-align: center; margin-top: 24px;">If you didn't request this code, please ignore this email or contact our support team if you have concerns about your account security.</p>
        </div>
        <div class="footer">
            <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
            <p style="margin-top: 12px;">This is an automated message. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
"""
    
    plain_text = f"""
PRISM - Email Verification

Hello,

You requested to {purpose_text}. Please use the following verification code:

OTP Code: {otp}

This code is valid for 10 minutes.

Enter this code in the verification form to proceed.

⚠️ Security Notice: Never share this code with anyone. PRISM staff will never ask for your OTP.

If you didn't request this code, please ignore this email or contact support if you have concerns.

© 2024 PRISM - APEXNEURAL
This is an automated message. Please do not reply to this email.
"""
    
    # OTP emails go to the user's email address from the form (user_data.email)
    # This is different from demo requests which go to a fixed recipient
    print(f"✅ [send_otp_email] Final confirmation - Sending to user's email: {email}")
    
    return send_mail(
        to_emails=email,  # User's email from form (user_data.email or email from forgot-password form)
        subject=subject,
        message=plain_text,
        password=password,
        from_email=from_email,
        html_content=html_content
    )


def send_email_with_attachment(to_email: str, subject: str, html_content: str, attachment_path: str = None, attachment_name: str = None):
    """Send email with single attachment"""
    from email.mime.base import MIMEBase
    from email import encoders
    
    from_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
    password = os.getenv("EMAIL_PASSWORD")
    
    if not from_email or not password:
        print(f"❌ Email credentials not found")
        return False
    
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    
    msg.attach(MIMEText(html_content, "html"))
    
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {attachment_name or os.path.basename(attachment_path)}"
            )
            msg.attach(part)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print(f"✅ Email with attachment sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Error sending email with attachment: {e}")
        return False


def send_email_with_attachments(to_email: str, subject: str, html_content: str, attachments: list):
    """Send email with multiple attachments"""
    from email.mime.base import MIMEBase
    from email import encoders
    
    from_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
    password = os.getenv("EMAIL_PASSWORD")
    
    if not from_email or not password:
        print(f"❌ Email credentials not found")
        return False
    
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    
    msg.attach(MIMEText(html_content, "html"))
    
    for attachment_path, attachment_name in attachments:
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {attachment_name}"
                )
                msg.attach(part)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print(f"✅ Email with attachments sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Error sending email with attachments: {e}")
        return False


async def send_interview_confirmation_emails(
    applicant_email: str,
    applicant_name: str,
    interviewer_emails: list,
    interview_date: str,
    interview_time: str,
    meeting_link: str,
    round_name: str,
    org_name: str,
    job_id: str,
    location_type: str = "online",
    location: str = None,
    feedback_form_link: str = None,
    jd_file_path: str = None,
    resume_file_path: str = None
) -> bool:
    """
    Send interview confirmation emails to applicant and interviewers
    """
    try:
        # Build conditional HTML parts
        meeting_link_html = ""
        location_html = ""
        join_button_html = ""
        instruction_text = "arrive at the location"
        
        if location_type == "online":
            meeting_link_html = f'<p><strong>Location:</strong> Online</p><p><strong>Meeting Link:</strong> <a href="{meeting_link}">{meeting_link}</a></p>'
            join_button_html = f'<div style="text-align: center;"><a href="{meeting_link}" class="button">Join Meeting</a></div>'
            instruction_text = "join the meeting"
        elif location_type == "offline":
            location_html = f'<p><strong>Location:</strong> Offline</p>'
            if location:
                location_html += f'<p><strong>Interview Address:</strong> {location}</p>'
            # For offline interviews, still include meeting link and feedback form
            if meeting_link:
                meeting_link_html = f'<p><strong>Meeting Link (Backup/Virtual Option):</strong> <a href="{meeting_link}">{meeting_link}</a></p>'
                join_button_html = f'<div style="text-align: center;"><a href="{meeting_link}" class="button">Join Meeting (Backup)</a></div>'
            instruction_text = "arrive at the location"
        
        # Send email to applicant
        applicant_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6; 
                    color: #111827;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 40px 20px;
                }}
                .container {{ 
                    max-width: 640px; 
                    margin: 0 auto; 
                    background: #ffffff;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 40px 32px; 
                    text-align: center;
                }}
                .header-icon {{
                    width: 70px;
                    height: 70px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 18px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 20px;
                    font-size: 36px;
                    backdrop-filter: blur(10px);
                }}
                .header h2 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 800;
                    letter-spacing: -0.5px;
                }}
                .content {{ 
                    background: #ffffff; 
                    padding: 36px 32px;
                }}
                .content p {{
                    margin: 0 0 16px 0;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #4b5563;
                }}
                .content p.greeting {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #111827;
                    margin-bottom: 20px;
                }}
                .details-box {{
                    background: linear-gradient(to bottom right, #f0f7ff, #e0f2fe);
                    padding: 24px;
                    border-radius: 16px;
                    margin: 28px 0;
                    border: 2px solid #667eea;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
                }}
                .details-box p {{
                    margin: 12px 0;
                    font-size: 15px;
                    color: #1f2937;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .details-box strong {{
                    font-weight: 700;
                    color: #111827;
                    min-width: 80px;
                }}
                .button {{ 
                    display: inline-block; 
                    padding: 16px 36px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    text-decoration: none; 
                    border-radius: 12px; 
                    margin: 24px 0;
                    font-weight: 700;
                    font-size: 16px;
                    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                    transition: all 0.3s;
                }}
                .button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
                }}
                .info-box {{
                    background: linear-gradient(to right, #d1fae5, #a7f3d0);
                    border-left: 4px solid #10b981;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 24px 0;
                }}
                .info-box p {{
                    margin: 0;
                    color: #065f46;
                    font-weight: 600;
                    font-size: 14px;
                }}
                .footer {{
                    background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
                    padding: 24px 32px;
                    text-align: center;
                    border-top: 2px solid #e5e7eb;
                }}
                .footer p {{
                    margin: 6px 0;
                    color: #6b7280;
                    font-size: 13px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-icon">📅</div>
                    <h2>Interview Scheduled</h2>
                    <p style="margin: 8px 0 0 0; opacity: 0.95; font-size: 15px;">Your interview has been confirmed</p>
                </div>
                <div class="content">
                    <p class="greeting">👋 Dear {applicant_name},</p>
                    <p>Great news! Your interview for <strong>{round_name}</strong> at <strong>{org_name}</strong> has been successfully scheduled.</p>
                    <div class="details-box">
                        <p><strong>📅 Date:</strong> {interview_date}</p>
                        <p><strong>⏰ Time:</strong> {interview_time}</p>
                        {location_html}
                        {meeting_link_html}
                    </div>
                    <div style="text-align: center;">
                        {join_button_html}
                    </div>
                    <div class="info-box">
                        <p>📎 Please {instruction_text} on time. The job description is attached for your reference.</p>
                    </div>
                    <p style="margin-top: 32px; padding-top: 24px; border-top: 2px solid #e5e7eb;">
                        <strong>Best regards,</strong><br>
                        <strong style="color: #667eea; font-size: 16px;">{org_name} Team</strong>
                    </p>
                </div>
                <div class="footer">
                    <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        send_email_with_attachment(
            to_email=applicant_email,
            subject=f"Interview Scheduled - {round_name} at {org_name}",
            html_content=applicant_html,
            attachment_path=jd_file_path,
            attachment_name="Job_Description.pdf"
        )
        
        # Build conditional HTML for interviewer email
        interviewer_meeting_link_html = ""
        interviewer_location_html = ""
        feedback_form_section = ""
        
        if location_type == "online" and meeting_link:
            interviewer_meeting_link_html = f'<p><strong>Location:</strong> Online</p><p><strong>Meeting Link:</strong> <a href="{meeting_link}">{meeting_link}</a></p>'
        elif location_type == "offline":
            interviewer_location_html = f'<p><strong>Location:</strong> Offline</p>'
            if location:
                interviewer_location_html += f'<p><strong>Interview Address:</strong> {location}</p>'
            # For offline interviews, still include meeting link
            if meeting_link:
                interviewer_meeting_link_html = f'<p><strong>Meeting Link (Backup/Virtual Option):</strong> <a href="{meeting_link}">{meeting_link}</a></p>'
        
        if feedback_form_link:
            feedback_form_section = f"""
                    <div class="feedback-section">
                        <p style="margin: 0 0 12px 0; font-size: 16px; font-weight: 700; color: #1e40af;">📝 Interview Feedback Required</p>
                        <p style="margin: 0 0 16px 0; color: #1e40af; font-size: 14px;">After conducting the interview, please fill out the feedback form to complete the evaluation process:</p>
                        <div style="text-align: center;">
                            <a href="{feedback_form_link}" class="feedback-button">📋 Fill Feedback Form</a>
                        </div>
                        <p style="margin: 16px 0 0 0; font-size: 12px; color: #1e40af; text-align: center; font-weight: 500;"><strong>Note:</strong> This form can only be submitted once. Please ensure all information is accurate before submitting.</p>
                    </div>
            """
        
        # Send email to interviewers
        interviewer_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6; 
                    color: #111827;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 40px 20px;
                }}
                .container {{ 
                    max-width: 640px; 
                    margin: 0 auto; 
                    background: #ffffff;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 40px 32px; 
                    text-align: center;
                }}
                .header-icon {{
                    width: 70px;
                    height: 70px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 18px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 20px;
                    font-size: 36px;
                    backdrop-filter: blur(10px);
                }}
                .header h2 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 800;
                    letter-spacing: -0.5px;
                }}
                .content {{ 
                    background: #ffffff; 
                    padding: 36px 32px;
                }}
                .content p {{
                    margin: 0 0 16px 0;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #4b5563;
                }}
                .content p.greeting {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #111827;
                    margin-bottom: 20px;
                }}
                .details-box {{
                    background: linear-gradient(to bottom right, #f0f7ff, #e0f2fe);
                    padding: 24px;
                    border-radius: 16px;
                    margin: 28px 0;
                    border: 2px solid #667eea;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
                }}
                .details-box p {{
                    margin: 12px 0;
                    font-size: 15px;
                    color: #1f2937;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .details-box strong {{
                    font-weight: 700;
                    color: #111827;
                    min-width: 100px;
                }}
                .feedback-section {{
                    background: linear-gradient(to right, #e0f2fe, #dbeafe);
                    border-left: 4px solid #3b82f6;
                    padding: 20px 24px;
                    border-radius: 12px;
                    margin: 24px 0;
                    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
                }}
                .feedback-button {{
                    display: inline-block;
                    padding: 14px 32px;
                    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 12px;
                    font-weight: 700;
                    font-size: 15px;
                    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
                    transition: all 0.3s;
                    text-align: center;
                    margin: 12px 0;
                }}
                .feedback-button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
                }}
                .attachments-note {{
                    background: linear-gradient(to right, #f0fdf4, #dcfce7);
                    border-left: 4px solid #22c55e;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 24px 0;
                }}
                .attachments-note p {{
                    margin: 0;
                    color: #166534;
                    font-weight: 600;
                    font-size: 14px;
                }}
                .footer {{
                    background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
                    padding: 24px 32px;
                    text-align: center;
                    border-top: 2px solid #e5e7eb;
                }}
                .footer p {{
                    margin: 6px 0;
                    color: #6b7280;
                    font-size: 13px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-icon">👨‍💼</div>
                    <h2>Interview Scheduled</h2>
                    <p style="margin: 8px 0 0 0; opacity: 0.95; font-size: 15px;">You've been assigned as an interviewer</p>
                </div>
                <div class="content">
                    <p class="greeting">👋 Hello,</p>
                    <p>An interview has been scheduled and assigned to you for review. Please find the details below:</p>
                    <div class="details-box">
                        <p><strong>👤 Candidate:</strong> {applicant_name}</p>
                        <p><strong>🎯 Round:</strong> {round_name}</p>
                        <p><strong>📅 Date:</strong> {interview_date}</p>
                        <p><strong>⏰ Time:</strong> {interview_time}</p>
                        {interviewer_location_html}
                        {interviewer_meeting_link_html}
                    </div>
                    {feedback_form_section}
                    <div class="attachments-note">
                        <p>📎 The candidate's resume and job description are attached for your review.</p>
                    </div>
                    <p style="margin-top: 32px; padding-top: 24px; border-top: 2px solid #e5e7eb;">
                        <strong>Best regards,</strong><br>
                        <strong style="color: #667eea; font-size: 16px;">{org_name} Team</strong>
                    </p>
                </div>
                <div class="footer">
                    <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Send to each interviewer
        attachments = []
        if jd_file_path:
            attachments.append((jd_file_path, "Job_Description.pdf"))
        if resume_file_path:
            attachments.append((resume_file_path, f"{applicant_name}_Resume.pdf"))
        
        for interviewer_email in interviewer_emails:
            send_email_with_attachments(
                to_email=interviewer_email,
                subject=f"Interview Scheduled - {applicant_name} - {round_name}",
                html_content=interviewer_html,
                attachments=attachments
            )
        
        return True
        
    except Exception as e:
        print(f"❌ Error sending interview confirmation emails: {e}")
        import traceback
        traceback.print_exc()
        return False


async def send_no_slots_notification_email(
    org_email: str,
    org_name: str,
    team_name: str
) -> bool:
    """
    Send email to organization when no interview slots are available
    """
    try:
        from_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
        password = os.getenv("EMAIL_PASSWORD")
        
        if not from_email or not password:
            print(f"❌ Email credentials not found")
            return False
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6; 
                    color: #111827;
                    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
                    margin: 0;
                    padding: 40px 20px;
                }}
                .container {{ 
                    max-width: 640px; 
                    margin: 0 auto; 
                    background: #ffffff;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
                    color: white; 
                    padding: 40px 32px; 
                    text-align: center;
                }}
                .header-icon {{
                    width: 80px;
                    height: 80px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 20px;
                    font-size: 42px;
                    backdrop-filter: blur(10px);
                }}
                .header h2 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 800;
                    letter-spacing: -0.5px;
                }}
                .content {{ 
                    background: #ffffff; 
                    padding: 36px 32px;
                }}
                .content p {{
                    margin: 0 0 16px 0;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #4b5563;
                }}
                .content p.greeting {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #111827;
                    margin-bottom: 20px;
                }}
                .alert-box {{ 
                    background: linear-gradient(to right, #fee2e2, #fecaca);
                    border-left: 4px solid #dc2626; 
                    padding: 20px 24px; 
                    margin: 24px 0;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.1);
                }}
                .alert-box p {{
                    margin: 8px 0;
                    color: #991b1b;
                    font-size: 15px;
                }}
                .alert-box strong {{
                    font-weight: 700;
                    color: #7f1d1d;
                }}
                .recommendations {{
                    background: linear-gradient(to bottom right, #f0f7ff, #e0f2fe);
                    padding: 24px;
                    border-radius: 16px;
                    margin: 24px 0;
                    border: 2px solid #3b82f6;
                }}
                .recommendations h3 {{
                    margin: 0 0 16px 0;
                    font-size: 18px;
                    font-weight: 700;
                    color: #1e40af;
                }}
                .recommendations ul {{
                    margin: 0;
                    padding-left: 24px;
                    color: #1e40af;
                }}
                .recommendations li {{
                    margin: 10px 0;
                    font-size: 14px;
                    line-height: 1.6;
                    font-weight: 500;
                }}
                .info-note {{
                    background: linear-gradient(to right, #fef3c7, #fde68a);
                    border-left: 4px solid #f59e0b;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 24px 0;
                }}
                .info-note p {{
                    margin: 0;
                    color: #92400e;
                    font-weight: 600;
                    font-size: 13px;
                }}
                .footer {{
                    background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
                    padding: 24px 32px;
                    text-align: center;
                    border-top: 2px solid #e5e7eb;
                }}
                .footer p {{
                    margin: 6px 0;
                    color: #6b7280;
                    font-size: 13px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-icon">⚠️</div>
                    <h2>No Interview Slots Available</h2>
                    <p style="margin: 8px 0 0 0; opacity: 0.95; font-size: 15px;">All team members are currently busy</p>
                </div>
                <div class="content">
                    <p class="greeting">Dear {org_name} Team,</p>
                    <div class="alert-box">
                        <p><strong>🚫 Alert:</strong> All team members in <strong>{team_name}</strong> are currently busy for the next 5 working days.</p>
                        <p><strong>Status:</strong> No interview slots are available for scheduling at this time.</p>
                    </div>
                    <div class="recommendations">
                        <h3>💡 Recommended Actions:</h3>
                        <ul>
                            <li><strong>Add more team members</strong> to {team_name} to increase availability</li>
                            <li><strong>Wait and retry</strong> scheduling after some time when slots open up</li>
                            <li><strong>Manual coordination</strong> with applicants for alternative arrangements</li>
                        </ul>
                    </div>
                    <div class="info-note">
                        <p>ℹ️ This is an automated notification sent when the system detects no available interview slots.</p>
                    </div>
                    <p style="margin-top: 32px; padding-top: 24px; border-top: 2px solid #e5e7eb;">
                        <strong>Best regards,</strong><br>
                        <strong style="color: #667eea; font-size: 16px;">PRISM System</strong>
                    </p>
                </div>
                <div class="footer">
                    <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        plain_text = f"""
No Interview Slots Available

Dear {org_name} Team,

Alert: All team members in {team_name} are currently busy for the next 5 working days.
No interview slots are available for scheduling.

Recommendation:
- Add more team members to {team_name} to increase availability
- Or wait for some time and try scheduling again
- Or manually coordinate with applicants for alternative arrangements

This notification is sent automatically when the system detects no available slots.

Best regards,
PRISM System
"""
        
        subject = f"⚠️ No Interview Slots Available - {team_name}"
        
        return send_mail(
            to_emails=org_email,
            subject=subject,
            message=plain_text,
            password=password,
            from_email=from_email,
            html_content=html_body
        )
        
    except Exception as e:
        print(f"❌ Error sending no slots notification email: {e}")
        return False


async def send_interview_form_email(
    applicant_email: str,
    applicant_name: str,
    form_link: str,
    round_name: str,
    org_name: str
) -> bool:
    """
    Send interview scheduling form email to applicant
    
    Args:
        applicant_email: Applicant's email address
        applicant_name: Applicant's name
        form_link: Link to the scheduling form
        round_name: Name of the interview round
        org_name: Organization name
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        from_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
        password = os.getenv("EMAIL_PASSWORD")
        
        if not from_email or not password:
            print(f"❌ Email credentials not found")
            return False
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6; 
                    color: #111827;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 40px 20px;
                }}
                .container {{ 
                    max-width: 640px; 
                    margin: 0 auto; 
                    background: #ffffff;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 40px 32px; 
                    text-align: center;
                }}
                .header-icon {{
                    width: 80px;
                    height: 80px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 20px;
                    font-size: 42px;
                    backdrop-filter: blur(10px);
                }}
                .header h2 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 800;
                    letter-spacing: -0.5px;
                }}
                .content {{ 
                    background: #ffffff; 
                    padding: 40px 36px;
                }}
                .content p {{
                    margin: 0 0 18px 0;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #4b5563;
                }}
                .content p.greeting {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #111827;
                    margin-bottom: 20px;
                }}
                .congrats-box {{
                    background: linear-gradient(to right, #d1fae5, #a7f3d0);
                    border-left: 4px solid #10b981;
                    padding: 20px 24px;
                    border-radius: 12px;
                    margin: 24px 0;
                    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
                }}
                .congrats-box p {{
                    margin: 0;
                    color: #065f46;
                    font-size: 16px;
                    font-weight: 600;
                }}
                .button {{ 
                    display: inline-block; 
                    padding: 18px 40px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    text-decoration: none; 
                    border-radius: 14px; 
                    margin: 24px 0;
                    font-weight: 700;
                    font-size: 16px;
                    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
                    transition: all 0.3s;
                    text-align: center;
                    letter-spacing: 0.3px;
                }}
                .button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 10px 28px rgba(102, 126, 234, 0.5);
                }}
                .link-box {{
                    background: #f9fafb;
                    border: 2px dashed #cbd5e1;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 20px 0;
                    word-break: break-all;
                }}
                .link-box p {{
                    margin: 0;
                    color: #667eea;
                    font-size: 13px;
                    font-weight: 600;
                    text-align: center;
                }}
                .info-note {{
                    background: linear-gradient(to right, #dbeafe, #e0e7ff);
                    border-left: 4px solid #3b82f6;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 24px 0;
                }}
                .info-note p {{
                    margin: 0;
                    color: #1e40af;
                    font-weight: 500;
                    font-size: 14px;
                }}
                .footer {{ 
                    text-align: center; 
                    margin-top: 24px;
                    background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
                    padding: 24px 32px;
                    border-top: 2px solid #e5e7eb;
                }}
                .footer p {{
                    margin: 6px 0;
                    color: #6b7280;
                    font-size: 13px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-icon">📅</div>
                    <h2>Interview Scheduling</h2>
                    <p style="margin: 8px 0 0 0; opacity: 0.95; font-size: 15px;">Select your preferred time slot</p>
                </div>
                <div class="content">
                    <p class="greeting">👋 Dear {applicant_name},</p>
                    <div class="congrats-box">
                        <p>🎉 Congratulations! You have been selected for the <strong>{round_name}</strong> at <strong>{org_name}</strong>!</p>
                    </div>
                    <p>We're excited to move forward with your application. Please schedule your interview at your earliest convenience.</p>
                    <div class="info-note">
                        <p>📌 Click the button below to view available time slots and select a date and time that works best for you.</p>
                    </div>
                    <div style="text-align: center; margin: 32px 0;">
                        <a href="{form_link}" class="button">📅 Schedule Your Interview</a>
                    </div>
                    <p style="text-align: center; color: #6b7280; font-size: 14px; margin: 20px 0;">If the button doesn't work, copy and paste this link into your browser:</p>
                    <div class="link-box">
                        <p>{form_link}</p>
                    </div>
                    <p style="margin-top: 32px; padding-top: 24px; border-top: 2px solid #e5e7eb;">
                        <strong>Best regards,</strong><br>
                        <strong style="color: #667eea; font-size: 16px;">{org_name} Team</strong>
                    </p>
                </div>
                <div class="footer">
                    <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        plain_text = f"""
Interview Scheduling

Dear {applicant_name},

Congratulations! You have been selected for the {round_name} at {org_name}.

Please click the link below to schedule your interview by selecting a convenient date and time slot:

{form_link}

Best regards,
{org_name} Team
"""
        
        subject = f"Schedule Your {round_name} Interview - {org_name}"
        
        return send_mail(
            to_emails=applicant_email,
            subject=subject,
            message=plain_text,
            password=password,
            from_email=from_email,
            html_content=html_body
        )
        
    except Exception as e:
        print(f"❌ Error sending interview form email: {e}")
        return False


async def send_interview_feedback_form_email(
    interviewer_email: str,
    interviewer_name: str,
    applicant_name: str,
    applicant_email: str,
    round_name: str,
    interview_date: str,
    interview_time: str,
    feedback_form_link: str,
    job_id: str,
    location_type: str = "online",
    location: str = None,
    meeting_link: str = None,
    resume_file_path: str = None,
    jd_file_path: str = None
) -> bool:
    """
    Send interview feedback form link to interviewer with resume and JD
    """
    try:
        # Build conditional HTML parts for feedback email
        feedback_meeting_link_html = ""
        feedback_location_html = ""
        
        if location_type == "online" and meeting_link:
            feedback_meeting_link_html = f'<p style="margin: 5px 0;">Meeting Link: <a href="{meeting_link}">{meeting_link}</a></p>'
        elif location_type == "offline" and location:
            feedback_location_html = f'<p style="margin: 5px 0;">Interview Location: {location}</p>'
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6; 
                    color: #111827;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 40px 20px;
                }}
                .container {{ 
                    max-width: 640px; 
                    margin: 0 auto; 
                    background: #ffffff;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 40px 32px; 
                    text-align: center;
                }}
                .header-icon {{
                    width: 80px;
                    height: 80px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 20px;
                    font-size: 42px;
                    backdrop-filter: blur(10px);
                }}
                .header h2 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 800;
                    letter-spacing: -0.5px;
                }}
                .content {{ 
                    background: #ffffff; 
                    padding: 36px 32px;
                }}
                .content p {{
                    margin: 0 0 16px 0;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #4b5563;
                }}
                .content p.greeting {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #111827;
                    margin-bottom: 20px;
                }}
                .button {{ 
                    display: inline-block; 
                    padding: 16px 36px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    text-decoration: none; 
                    border-radius: 12px; 
                    margin: 24px 0;
                    font-weight: 700;
                    font-size: 16px;
                    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                    transition: all 0.3s;
                }}
                .button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
                }}
                .details-box {{ 
                    background: linear-gradient(to bottom right, #f0f7ff, #e0f2fe);
                    padding: 24px;
                    border-radius: 16px;
                    margin: 24px 0;
                    border: 2px solid #667eea;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
                }}
                .details-box h3 {{
                    margin: 0 0 16px 0;
                    font-size: 16px;
                    font-weight: 700;
                    color: #1e40af;
                }}
                .details-box p {{
                    margin: 10px 0;
                    font-size: 14px;
                    color: #1f2937;
                }}
                .details-box strong {{
                    font-weight: 700;
                    color: #111827;
                    min-width: 120px;
                    display: inline-block;
                }}
                .warning-note {{
                    background: linear-gradient(to right, #fef3c7, #fde68a);
                    border-left: 4px solid #f59e0b;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 24px 0;
                }}
                .warning-note p {{
                    margin: 0;
                    color: #92400e;
                    font-weight: 600;
                    font-size: 14px;
                }}
                .attachments-note {{
                    background: linear-gradient(to right, #d1fae5, #a7f3d0);
                    border-left: 4px solid #10b981;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 24px 0;
                }}
                .attachments-note p {{
                    margin: 0;
                    color: #065f46;
                    font-weight: 600;
                    font-size: 14px;
                }}
                .footer {{
                    background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
                    padding: 24px 32px;
                    text-align: center;
                    border-top: 2px solid #e5e7eb;
                }}
                .footer p {{
                    margin: 6px 0;
                    color: #6b7280;
                    font-size: 13px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-icon">📝</div>
                    <h2>Interview Feedback Required</h2>
                    <p style="margin: 8px 0 0 0; opacity: 0.95; font-size: 15px;">Please submit your evaluation</p>
                </div>
                <div class="content">
                    <p class="greeting">👋 Dear {interviewer_name},</p>
                    <p>You recently conducted an interview with <strong>{applicant_name}</strong> (<a href="mailto:{applicant_email}" style="color: #667eea; text-decoration: none;">{applicant_email}</a>) for the <strong>{round_name}</strong>.</p>
                    <div class="details-box">
                        <h3>📋 Interview Details</h3>
                        <p><strong>📅 Date:</strong> {interview_date}</p>
                        <p><strong>⏰ Time:</strong> {interview_time}</p>
                        <p><strong>🎯 Round:</strong> {round_name}</p>
                        <p><strong>📍 Location Type:</strong> {location_type.capitalize()}</p>
                        {feedback_meeting_link_html}
                        {feedback_location_html}
                    </div>
                    <p style="text-align: center; font-size: 16px; color: #1f2937; font-weight: 600; margin: 24px 0;">Please fill out the feedback form to complete the interview evaluation:</p>
                    <div style="text-align: center;">
                        <a href="{feedback_form_link}" class="button">📋 Fill Feedback Form</a>
                    </div>
                    <div class="warning-note">
                        <p>⚠️ <strong>Important:</strong> This form can only be submitted once. Please ensure all information is accurate before submitting.</p>
                    </div>
                    <div class="attachments-note">
                        <p>📎 The candidate's resume and job description are attached for your reference.</p>
                    </div>
                    <p style="margin-top: 32px; padding-top: 24px; border-top: 2px solid #e5e7eb;">
                        <strong>Best regards,</strong><br>
                        <strong style="color: #667eea; font-size: 16px;">PRISM System</strong>
                    </p>
                </div>
                <div class="footer">
                    <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        plain_text = f"""
Interview Feedback Required

Dear {interviewer_name},

You recently conducted an interview with {applicant_name} ({applicant_email}) for the {round_name}.

Interview Details:
Date: {interview_date}
Time: {interview_time}
Round: {round_name}

Please fill out the feedback form to complete the interview process:
{feedback_form_link}

Note: This form can only be submitted once. Please ensure all information is accurate before submitting.

The candidate's resume and job description are attached for your reference.

Best regards,
PRISM System
"""
        
        subject = f"Interview Feedback Form - {round_name} - {applicant_name}"
        
        # Prepare attachments
        attachments = []
        if jd_file_path and os.path.exists(jd_file_path):
            attachments.append((jd_file_path, f"Job_Description_{job_id}.pdf"))
        if resume_file_path and os.path.exists(resume_file_path):
            attachments.append((resume_file_path, f"Resume_{applicant_name.replace(' ', '_')}.pdf"))
        
        if attachments:
            send_email_with_attachments(
                to_email=interviewer_email,
                subject=subject,
                html_content=html_body,
                attachments=attachments
            )
        else:
            send_email_with_attachment(
                to_email=interviewer_email,
                subject=subject,
                html_content=html_body,
                attachment_path=None,
                attachment_name=None
            )
        
        print(f"✅ Feedback form email sent to {interviewer_email}")
        return True
        
    except Exception as e:
        print(f"❌ Error sending feedback form email: {e}")
        import traceback
        traceback.print_exc()
        return False


async def send_offer_letter_email(
    applicant_email: str,
    applicant_name: str,
    org_name: str,
    job_role: str,
    offer_letter_path: str,
    form_link: str
) -> bool:
    """
    Send offer letter email with attachment and accept/reject form link
    """
    try:
        from_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
        password = os.getenv("EMAIL_PASSWORD")
        
        if not from_email or not password:
            print(f"❌ Email credentials not found")
            return False
        
        # Read offer letter file
        with open(offer_letter_path, 'rb') as f:
            offer_letter_data = f.read()
            offer_letter_name = os.path.basename(offer_letter_path)
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6; 
                    color: #111827;
                    background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
                    margin: 0;
                    padding: 40px 20px;
                }}
                .container {{ 
                    max-width: 640px; 
                    margin: 0 auto; 
                    background: #ffffff;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
                    color: white; 
                    padding: 50px 32px; 
                    text-align: center;
                    position: relative;
                }}
                .header-icon {{
                    width: 100px;
                    height: 100px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 25px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 24px;
                    font-size: 56px;
                    backdrop-filter: blur(10px);
                    animation: celebrate 0.5s ease-in-out;
                }}
                @keyframes celebrate {{
                    0%, 100% {{ transform: scale(1) rotate(0deg); }}
                    25% {{ transform: scale(1.1) rotate(-5deg); }}
                    75% {{ transform: scale(1.1) rotate(5deg); }}
                }}
                .header h2 {{
                    margin: 0 0 12px 0;
                    font-size: 36px;
                    font-weight: 800;
                    letter-spacing: -1px;
                }}
                .header p {{
                    margin: 0;
                    font-size: 16px;
                    opacity: 0.95;
                    font-weight: 500;
                }}
                .content {{ 
                    background: #ffffff; 
                    padding: 40px 36px;
                }}
                .content p {{
                    margin: 0 0 18px 0;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #4b5563;
                }}
                .content p.greeting {{
                    font-size: 17px;
                    font-weight: 600;
                    color: #111827;
                    margin-bottom: 24px;
                }}
                .success-box {{
                    background: linear-gradient(to right, #d1fae5, #a7f3d0);
                    border: 2px solid #10b981;
                    padding: 24px;
                    border-radius: 16px;
                    margin: 28px 0;
                    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
                    text-align: center;
                }}
                .success-box p {{
                    margin: 0;
                    color: #065f46;
                    font-size: 17px;
                    font-weight: 700;
                    line-height: 1.6;
                }}
                .button-group {{
                    text-align: center;
                    margin: 36px 0;
                    display: block;
                }}
                .button {{ 
                    display: block; 
                    width: 100%;
                    max-width: 300px;
                    margin: 0 auto 16px auto;
                    padding: 18px 40px; 
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white; 
                    text-decoration: none; 
                    border-radius: 14px; 
                    font-weight: 700;
                    font-size: 16px;
                    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
                    transition: all 0.3s;
                    letter-spacing: 0.3px;
                    text-align: center;
                }}
                .button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 10px 28px rgba(16, 185, 129, 0.5);
                }}
                .button-reject {{ 
                    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4);
                    margin-bottom: 0;
                }}
                .button-reject:hover {{
                    box-shadow: 0 10px 28px rgba(239, 68, 68, 0.5);
                }}
                .attachment-note {{
                    background: linear-gradient(to right, #fef3c7, #fde68a);
                    border-left: 4px solid #f59e0b;
                    padding: 18px 20px;
                    border-radius: 12px;
                    margin: 24px 0;
                }}
                .attachment-note p {{
                    margin: 0;
                    color: #92400e;
                    font-weight: 600;
                    font-size: 14px;
                }}
                .link-box {{
                    background: #f9fafb;
                    border: 2px dashed #cbd5e1;
                    padding: 16px 20px;
                    border-radius: 12px;
                    margin: 20px 0;
                    word-break: break-all;
                }}
                .link-box p {{
                    margin: 0;
                    color: #10b981;
                    font-size: 13px;
                    font-weight: 600;
                    text-align: center;
                }}
                .footer {{ 
                    text-align: center;
                    background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
                    padding: 28px 32px;
                    border-top: 2px solid #e5e7eb;
                }}
                .footer p {{
                    margin: 6px 0;
                    color: #6b7280;
                    font-size: 13px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-icon">🎉</div>
                    <h2>Congratulations!</h2>
                    <p>You've been selected for the position</p>
                </div>
                <div class="content">
                    <p class="greeting">👋 Dear {applicant_name},</p>
                    <div class="success-box">
                        <p>✨ We are thrilled to inform you that you have been <strong>selected for the role of {job_role}</strong> at <strong>{org_name}</strong>! ✨</p>
                    </div>
                    <p style="text-align: center; font-size: 16px; color: #1f2937;">This is an exciting milestone in your career journey, and we're delighted to have you join our team!</p>
                    <div class="attachment-note">
                        <p>📎 <strong>Your offer letter is attached</strong> to this email with all the details about your role, compensation, and benefits.</p>
                    </div>
                    <p style="text-align: center; font-size: 16px; color: #1f2937; font-weight: 600; margin: 28px 0 20px 0;">Please review the offer and let us know your decision:</p>
                    <div class="button-group">
                        <a href="{form_link}&response=accept" class="button">✅ Accept Offer</a>
                        <a href="{form_link}&response=reject" class="button button-reject">❌ Reject Offer</a>
                    </div>
                    <p style="text-align: center; color: #6b7280; font-size: 14px; margin: 24px 0 16px 0;">If the buttons don't work, copy and paste this link into your browser:</p>
                    <div class="link-box">
                        <p>{form_link}</p>
                    </div>
                    <p style="margin-top: 36px; padding-top: 28px; border-top: 2px solid #e5e7eb; text-align: center;">
                        <strong style="display: block; margin-bottom: 8px;">We're excited to welcome you aboard!</strong>
                        <strong style="color: #10b981; font-size: 18px; display: block; margin-top: 12px;">{org_name} Team</strong>
                    </p>
                </div>
                <div class="footer">
                    <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
                    <p style="margin-top: 12px;">This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create a concise plain text version (for email clients that don't support HTML)
        plain_text = f"""Congratulations!

Dear {applicant_name},

We are thrilled to inform you that you have been selected for the role of {job_role} at {org_name}!

Your offer letter is attached to this email with all the details about your role, compensation, and benefits.

Please review the offer and let us know your decision by visiting:
{form_link}

We're excited to welcome you aboard!

Best regards,
{org_name} Team

---
© 2024 PRISM - APEXNEURAL
This is an automated message. Please do not reply to this email.
"""
        
        subject = f"🎉 Offer Letter - {org_name}"
        
        # Create email with multipart/alternative for proper email client handling
        msg = MIMEMultipart('mixed')
        msg['From'] = from_email
        msg['To'] = applicant_email
        msg['Subject'] = subject
        
        # Create multipart/alternative container for text and HTML
        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)
        
        # Attach plain text and HTML as alternatives (email client will choose one)
        msg_alternative.attach(MIMEText(plain_text, 'plain'))
        msg_alternative.attach(MIMEText(html_body, 'html'))
        
        # Attach offer letter
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(offer_letter_data)
        encoders.encode_base64(attachment)
        attachment.add_header(
            'Content-Disposition',
            f'attachment; filename= {offer_letter_name}'
        )
        msg.attach(attachment)
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Offer letter email sent to {applicant_email}")
        return True
        
    except Exception as e:
        print(f"❌ Error sending offer letter email: {e}")
        import traceback
        traceback.print_exc()
        return False