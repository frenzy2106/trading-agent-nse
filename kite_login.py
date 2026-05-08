"""
Kite Connect auth flow.

Run this script once a day (tokens expire at 6 AM IST):
    python kite_login.py

It will:
1. Print a login URL — open it in your browser
2. After login, Kite redirects to http://127.0.0.1?request_token=XXXX (page won't load — that's fine)
3. Paste the full redirect URL here
4. Script exchanges the request_token for an access_token and saves it to .kite_session.json
"""

import json
import os
import sys
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from kiteconnect import KiteConnect

SESSION_FILE = ".kite_session.json"


def load_credentials():
    load_dotenv()
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    if not api_key or not api_secret:
        sys.exit("ERROR: KITE_API_KEY and KITE_API_SECRET must be set in .env")
    return api_key, api_secret


def login():
    api_key, api_secret = load_credentials()
    kite = KiteConnect(api_key=api_key)

    print("\n--- Kite Login ---")
    print("1. Open this URL in your browser:\n")
    print(f"   {kite.login_url()}\n")
    print("2. Log in with your Zerodha credentials.")
    print("3. After login, copy the full URL from your browser address bar.")
    print("   (It starts with http://127.0.0.1?request_token=...)\n")

    redirect_url = input("Paste the redirect URL here: ").strip()

    try:
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        request_token = params["request_token"][0]
    except (KeyError, IndexError):
        sys.exit("ERROR: Could not find request_token in the URL. Did you paste the full URL?")

    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]

    session = {
        "access_token": access_token,
        "api_key": api_key,
        "login_time": datetime.now().isoformat(),
    }
    with open(SESSION_FILE, "w") as f:
        json.dump(session, f, indent=2)

    print(f"\nSuccess! Access token saved to {SESSION_FILE}")
    print("This token expires at 6 AM IST tomorrow — re-run this script each morning.\n")
    return kite, access_token


def get_kite_client() -> KiteConnect:
    """Load a ready-to-use KiteConnect client from cached session. Raises if session missing."""
    if not os.path.exists(SESSION_FILE):
        sys.exit(f"ERROR: No session found. Run `python kite_login.py` first.")

    with open(SESSION_FILE) as f:
        session = json.load(f)

    kite = KiteConnect(api_key=session["api_key"])
    kite.set_access_token(session["access_token"])
    return kite


if __name__ == "__main__":
    login()
