# Authentication setup

The application uses Streamlit's native OpenID Connect support. Google or
ORCID authenticates the user; the application never receives a password.

## 1. Choose the callback URL

Every provider must use the same callback URL:

- Local: `http://localhost:8501/oauth2callback`
- Streamlit Community Cloud: `https://YOUR-APP.streamlit.app/oauth2callback`
- Own server: `https://YOUR-DOMAIN/oauth2callback`

Production deployments must use HTTPS.

## 2. Create provider credentials

### Google

Create an OAuth 2.0 client for a **Web application** in Google Cloud and add
the callback URL as an authorized redirect URI. If the consent screen is in
testing mode, add every account that should be able to test the application.

Google documentation:
<https://developers.google.com/identity/openid-connect/openid-connect>

### ORCID

Register a Public API client and add the same callback URL. Use the ORCID
sandbox first if the integration is still under development, then register
production credentials before release.

ORCID documentation:
<https://info.orcid.org/documentation/api-tutorials/api-tutorial-get-and-authenticated-orcid-id/>

For sandbox testing, change the ORCID metadata URL in the example secrets to:

```toml
server_metadata_url = "https://sandbox.orcid.org/.well-known/openid-configuration"
```

## 3. Configure secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, replace
all placeholders, and generate a strong cookie secret, for example:

```bash
openssl rand -hex 32
```

Never commit `.streamlit/secrets.toml`. It is ignored by Git.

On Streamlit Community Cloud, paste the same TOML into **App settings →
Secrets**, using the deployed callback URL. On a private server, provide the
same configuration through the deployment's secret-management mechanism.

## 4. Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The identity key is the combination of the verified OIDC `iss` (issuer) and
`sub` (subject) claims. ORCID may not return an email claim, so those users are
asked for a contact email in the research profile. This contact field is not
used as proof of identity. Research-profile fields currently remain only in
the active Streamlit session; a production database can persist them later
without changing the authentication flow.

Uploaded files are automatically copied byte-for-byte to `data/uploads/`.
Their backup names contain the upload date, uploader email, and original
filename. Before writing, the application compares SHA-256 content hashes with
existing files. If identical content already exists, the original is kept and
is not overwritten. Protect this directory with appropriate filesystem access
controls and include it in the server backup policy.
