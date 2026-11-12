# Anonymous access

The maintained Streamlit application opens directly, without registration,
researcher profiles, Google/ORCID login, SMTP or authentication secrets.

Uploads and results remain in session memory. Download results before closing
the session. No new uploads are archived on disk. Previously archived files
are not deleted by this update; the server administrator should preserve or
remove those under the institution's existing retention policy.

Old OAuth/SMTP service variables and `.streamlit/secrets.toml` are unnecessary
for this application. Do not commit secrets. Remove obsolete service settings
only after confirming they are not shared with another application.
