import os
import sqlite3
from datetime import datetime
import csv

# Paths
db_file = "attendance.db"
csv_path = "attendance.csv"

# Setup DB table if needed
def setup_db():
    try:
        conn = sqlite3.connect(db_file)
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS attendance ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "total_faces INTEGER, "
                "real_faces INTEGER, "
                "timestamp TEXT)"
            )
    except sqlite3.DatabaseError as db_err:
        print("DB init error:", db_err)

# Add attendance to DB and CSV
def record_attendance(seen, confirmed):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = sqlite3.connect(db_file)
        with conn:
            conn.execute(
                "INSERT INTO attendance (total_faces, real_faces, timestamp) "
                "VALUES (?, ?, ?)", (seen, confirmed, now)
            )
    except sqlite3.OperationalError as op_err:
        print("Insert failed:", op_err)
        return

    try:
        new_csv = not os.path.exists(csv_path)
        f = open(csv_path, 'a', newline='')
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Total", "Confirmed"])

        if new_csv:
            writer.writeheader()

        writer.writerow({
            "Timestamp": now,
            "Total": seen,
            "Confirmed": confirmed
        })
        f.close()
    except Exception as e:
        print("CSV error:", str(e))
