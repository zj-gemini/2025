import sqlite3
import os

DB_FILE = "tickets.db"


def get_db_connection():
    """Creates a database connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database, overwriting it if it already exists."""
    # Always overwrite the database file on startup for a clean slate in development.
    if os.path.exists(DB_FILE):
        print(f"Removing existing database file: {DB_FILE}")
        os.remove(DB_FILE)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_question TEXT NOT NULL,
            error_description TEXT,
            conversation_history TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_contact TEXT,
            status TEXT DEFAULT 'open'
        )
    """
    )
    # Insert dummy data
    cursor.execute(
        "INSERT INTO tickets (user_question, user_contact, error_description) VALUES (?, ?, ?)",
        ("My internet is not working.", "user1@example.com", "Manual entry"),
    )
    cursor.execute(
        "INSERT INTO tickets (user_question, user_contact, error_description) VALUES (?, ?, ?)",
        ("I can't access my account.", "user2@example.com", "Manual entry"),
    )
    conn.commit()
    conn.close()


def get_all_tickets():
    """Retrieves all tickets from the database."""
    conn = get_db_connection()
    tickets = conn.execute("SELECT * FROM tickets").fetchall()
    conn.close()
    return [dict(ticket) for ticket in tickets]


def create_ticket(
    user_question: str,
    user_contact: str,
    conversation_history: str,
    error_description: str,
):
    """Creates a new ticket in the database."""
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO tickets (user_question, user_contact, conversation_history, error_description) VALUES (?, ?, ?, ?)",
        (user_question, user_contact, conversation_history, error_description),
    )
    conn.commit()
    conn.close()
