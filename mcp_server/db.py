"""SQLite storage for the helpdesk fixture data.

The database is created from seed.json the first time it is opened. Set HELPDESK_DB to
point at a different file (the tests do this so they never touch the real fixture).
"""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED_FILE = DATA_DIR / "seed.json"
DEFAULT_DB = DATA_DIR / "helpdesk.db"

SCHEMA = """
CREATE TABLE kb_articles (id TEXT PRIMARY KEY, category TEXT, title TEXT, body TEXT);
CREATE TABLE tickets (
    id TEXT PRIMARY KEY, category TEXT, severity TEXT, status TEXT, title TEXT, description TEXT
);
CREATE TABLE ticket_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT REFERENCES tickets(id), note TEXT
);
"""


def db_path() -> Path:
    return Path(os.environ.get("HELPDESK_DB", DEFAULT_DB))


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open the database (seeding it on first use), commit on success, always close."""
    path = db_path()
    is_new = not path.exists()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        if is_new:
            _seed(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _seed(conn: sqlite3.Connection) -> None:
    seed = json.loads(SEED_FILE.read_text())
    conn.executescript(SCHEMA)
    for a in seed["kb_articles"]:
        conn.execute(
            "INSERT INTO kb_articles VALUES (?, ?, ?, ?)",
            (a["id"], a["category"], a["title"], a["body"]),
        )
    for t in seed["tickets"]:
        conn.execute(
            "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?)",
            (t["id"], t["category"], t["severity"], t["status"], t["title"], t["description"]),
        )
        for note in t["notes"]:
            conn.execute(
                "INSERT INTO ticket_notes (ticket_id, note) VALUES (?, ?)", (t["id"], note)
            )
    conn.commit()
