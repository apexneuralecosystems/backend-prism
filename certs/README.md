# Local HTTPS certificates

Place your PEM key and cert here so the backend can run on `https://localhost:5555`.

## Option 1: mkcert (recommended, trusted by browser)

1. Install [mkcert](https://github.com/FiloSottile/mkcert): `choco install mkcert` (Windows) or `brew install mkcert` (Mac).
2. Install local CA: `mkcert -install`
3. Generate certs in this folder:
   ```bash
   cd backend/certs
   mkcert -key-file localhost-key.pem -cert-file localhost.pem localhost 127.0.0.1 ::1
   ```
4. In `.env` set:
   - `SSL_KEYFILE=certs/localhost-key.pem`
   - `SSL_CERTFILE=certs/localhost.pem`

## Option 2: OpenSSL (self-signed, browser will show warning)

```bash
cd backend/certs
openssl req -x509 -newkey rsa:4096 -keyout localhost-key.pem -out localhost.pem -days 365 -nodes -subj "/CN=localhost"
```

Then in Chrome go to `https://localhost:5555`, accept the security warning (Advanced → Proceed). LinkedIn redirect will work.

## After adding certs

Restart the backend: `python main.py`. You should see Uvicorn serving over HTTPS on port 5555.
