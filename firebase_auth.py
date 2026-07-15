"""
firebase_auth.py
-----------------
Verifies Firebase ID tokens sent by the frontend (Authorization: Bearer <token>)
using pyrebase (a thin wrapper around Firebase's REST API), so only signed-in
users can call the /api/* routes, and so one user can't guess another user's
session_id.

Unlike firebase-admin, pyrebase can't verify a token's JWT signature locally —
it calls Firebase's `getAccountInfo` REST endpoint with the token on every
request. If the token is valid, Firebase returns the account (including the
uid); if it's expired/forged, the call fails. Slightly slower than local
verification, no service-account JSON needed though — just the same public
web config values you already use on the frontend.

Configure via env vars (values come straight from your Firebase web app
config — the same object you pasted into templates/index.html):

    FIREBASE_API_KEY        (required — this is what turns auth checks on)
    FIREBASE_AUTH_DOMAIN    (optional)
    FIREBASE_DATABASE_URL   (optional — pyrebase wants the key even if unused)
    FIREBASE_STORAGE_BUCKET (optional)

If FIREBASE_API_KEY isn't set, auth checks are silently disabled (g.uid stays
None) so local development works without any Firebase setup at all.
"""

import functools
import os

from dotenv import load_dotenv
from flask import g, jsonify, request

load_dotenv()  # reads .env so FIREBASE_* vars are available even if app.py didn't load it first

_auth_client = None
_enabled = False


def init_firebase():
    """Call once at app startup."""
    global _auth_client, _enabled

    api_key = os.environ.get("FIREBASE_API_KEY")
    if not api_key:
        print(
            "[firebase_auth] FIREBASE_API_KEY is not set — API routes are NOT "
            "auth-protected (dev mode). Set that env var in production to "
            "require sign-in."
        )
        return

    try:
        import pyrebase

        config = {
            "apiKey": api_key,
            "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
            "databaseURL": os.environ.get("FIREBASE_DATABASE_URL", ""),
            "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        }
        firebase = pyrebase.initialize_app(config)
        _auth_client = firebase.auth()
        _enabled = True
        print("[firebase_auth] pyrebase initialized — API routes now require sign-in.")
    except Exception as e:
        print(f"[firebase_auth] Failed to initialize ({e}). API routes are NOT auth-protected.")


def is_enabled():
    return _enabled


def require_auth(fn):
    """Route decorator. When Firebase isn't configured this is a no-op
    (g.uid = None). When it is configured, rejects requests without a valid
    Firebase ID token and sets g.uid to the verified user's uid."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _enabled:
            g.uid = None
            return fn(*args, **kwargs)

        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Please sign in to use AutoAI."}), 401

        id_token = header.split(" ", 1)[1]
        try:
            account_info = _auth_client.get_account_info(id_token)
            g.uid = account_info["users"][0]["localId"]
        except Exception:
            return jsonify({"error": "Your session expired — please sign in again."}), 401

        return fn(*args, **kwargs)

    return wrapper