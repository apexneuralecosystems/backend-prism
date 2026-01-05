# PRISM Backend API - Unified FastAPI Backend

**Complete backend solution** with authentication, user profiles, organization profiles, file uploads, and demo requests.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy `.env.example` to `.env` in the backend directory
   - Fill in all required values:
     ```env
     MONGO_URL=mongodb://localhost:27017/prism_db
     JWT_ACCESS_SECRET=your_access_secret_here
     JWT_REFRESH_SECRET=your_refresh_secret_here
     FROM_EMAIL=your-email@gmail.com
     EMAIL_PASSWORD=your-app-password
     TO_EMAIL=recipient-email@gmail.com
     SMTP_SERVER=smtp.gmail.com
     SMTP_PORT=587
     CORS_ORIGINS=http://localhost:3000,http://localhost:5173
     ```
   - **Important**: Use strong, random strings for JWT secrets
   - For CORS_ORIGINS, add all frontend URLs that will access the API (comma-separated)

3. **Run the server:**
   ```bash
   python main.py
   ```
   
   Server runs on: **http://localhost:8000**
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **View API Documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📋 API Endpoints

### Authentication (`/api/auth/`)

- **POST `/api/auth/signup`** - Send OTP for signup
- **POST `/api/auth/verify-otp`** - Verify OTP and create account
- **POST `/api/auth/login`** - User/Organization login
- **POST `/api/auth/logout`** - Logout and invalidate refresh token
- **POST `/api/auth/forgot-password`** - Send OTP for password reset
- **POST `/api/auth/reset-password`** - Reset password with OTP
- **POST `/api/auth/refresh-token`** - Refresh access token

### User Profiles (`/api/user-profile/`)

- **POST `/api/user-profile`** - Create/Update user profile (Protected)
- **GET `/api/user-profile`** - Get user profile (Protected)

### Organization Profiles (`/api/organization-profile/`)

- **POST `/api/organization-profile`** - Create/Update org profile (Protected)
- **GET `/api/organization-profile`** - Get org profile (Protected)

### File Uploads (`/api/upload/`)

- **POST `/api/upload`** - Upload files (resumes, photos, logos)

### Demo Requests (`/api/demo-request/`)

- **POST `/api/demo-request`** - Submit demo request and send email

### Health Check

- **GET `/health`** - Check API health status

## 🔐 Authentication Flow

1. **Signup**: User signs up → OTP sent to email
2. **Verify OTP**: User verifies OTP → Account created + tokens returned
3. **Login**: User logs in → Tokens returned
4. **Protected Routes**: Use `Authorization: Bearer <access_token>` header
5. **Token Refresh**: Use refresh token to get new access token when expired

## 📚 Documentation

- **`HOW_TO_USE.md`** - Detailed API usage guide with examples
- **`QUICKSTART.md`** - Quick start guide
- **`CONSOLIDATION.md`** - Backend consolidation information

## 🗄️ Database (MongoDB)

Collections:
- `users` - User accounts (both users and organizations)
- `user_data` - User profile data
- `organization_data` - Organization profile data
- `refresh_tokens` - Refresh token storage
- `otps` - OTP storage with TTL (10 minutes)

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- OTP verification for signup and password reset
- Refresh token rotation
- CORS protection
- Input validation

## 📧 Email Configuration

For Gmail:
1. Enable 2-Step Verification
2. Generate an App Password
3. Use the App Password in `EMAIL_PASSWORD`

## 🌐 CORS

CORS origins are configured via the `CORS_ORIGINS` environment variable (comma-separated list).

Default: `http://localhost:3000,http://localhost:5173`

To allow requests from other IPs, add them to `CORS_ORIGINS` in your `.env` file:
```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://192.168.1.100:5173
```

## ✅ Features

- ✅ JWT Authentication (Access + Refresh tokens)
- ✅ OTP-based Email Verification
- ✅ Password Reset with OTP
- ✅ User Profile Management
- ✅ Organization Profile Management
- ✅ File Uploads (Resumes, Photos, Logos)
- ✅ Demo Request System
- ✅ Email Notifications
- ✅ MongoDB Database
- ✅ Protected Routes
- ✅ Token Refresh

## 📝 Notes

- This is a **unified backend** - all features in one FastAPI application
- The old `auth-backend/` (Node.js) is no longer needed
- See `CONSOLIDATION.md` for migration details




