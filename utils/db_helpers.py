import sqlite3
import os
import json
from datetime import datetime

DEFAULT_DB_PATH = "traffic_violations.db"

def init_db(db_path=DEFAULT_DB_PATH):
    """Initializes the SQLite database and creates the violations table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            junction TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            license_plate TEXT NOT NULL,
            violations TEXT NOT NULL, -- JSON string or comma-separated list of violations
            confidence REAL NOT NULL,
            image_path TEXT,
            pdf_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_violation(junction, vehicle_type, license_plate, violations_list, confidence, image_path=None, pdf_path=None, db_path=DEFAULT_DB_PATH):
    """Logs a new violation record to the database."""
    init_db(db_path) # Ensure DB and table are initialized
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    violations_str = json.dumps(violations_list)
    
    cursor.execute("""
        INSERT INTO violations (timestamp, junction, vehicle_type, license_plate, violations, confidence, image_path, pdf_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, junction, vehicle_type, license_plate, violations_str, confidence, image_path, pdf_path))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id

def get_all_records(db_path=DEFAULT_DB_PATH):
    """Retrieves all violation records from the database, sorted by timestamp descending."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for r in rows:
        record = dict(r)
        # Decode JSON list
        try:
            record["violations_list"] = json.loads(record["violations"])
        except Exception:
            record["violations_list"] = [record["violations"]]
        records.append(record)
    return records

def search_records(search_query="", filter_field="All", db_path=DEFAULT_DB_PATH):
    """Searches and filters violation records based on license plate, junction, or violation type."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM violations WHERE 1=1"
    params = []
    
    if search_query:
        if filter_field == "License Plate":
            query += " AND license_plate LIKE ?"
            params.append(f"%{search_query}%")
        elif filter_field == "Junction":
            query += " AND junction LIKE ?"
            params.append(f"%{search_query}%")
        elif filter_field == "Violation Type":
            query += " AND violations LIKE ?"
            params.append(f"%{search_query}%")
        else: # All fields
            query += " AND (license_plate LIKE ? OR junction LIKE ? OR violations LIKE ? OR vehicle_type LIKE ?)"
            params.extend([f"%{search_query}%"] * 4)
            
    query += " ORDER BY timestamp DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for r in rows:
        record = dict(r)
        try:
            record["violations_list"] = json.loads(record["violations"])
        except Exception:
            record["violations_list"] = [record["violations"]]
        records.append(record)
    return records

def get_analytics_summary(db_path=DEFAULT_DB_PATH):
    """Computes violation aggregations and metrics for charts and dashboards."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Total count
    cursor.execute("SELECT COUNT(*) FROM violations")
    total_violations = cursor.fetchone()[0]
    
    # 2. Busiest junctions
    cursor.execute("SELECT junction, COUNT(*) FROM violations GROUP BY junction ORDER BY COUNT(*) DESC LIMIT 5")
    junctions = cursor.fetchall()
    
    # 3. Vehicle distribution
    cursor.execute("SELECT vehicle_type, COUNT(*) FROM violations GROUP BY vehicle_type")
    vehicles = cursor.fetchall()
    
    # 4. Violations breakdown (need to parse JSON lists)
    cursor.execute("SELECT violations FROM violations")
    all_violations_raw = cursor.fetchall()
    
    violation_counts = {}
    for (v_raw,) in all_violations_raw:
        try:
            v_list = json.loads(v_raw)
            for v in v_list:
                violation_counts[v] = violation_counts.get(v, 0) + 1
        except Exception:
            # Fallback if raw text
            violation_counts[v_raw] = violation_counts.get(v_raw, 0) + 1
            
    # 5. Over time (Hourly/Daily)
    cursor.execute("SELECT strftime('%Y-%m-%d', timestamp) as day, COUNT(*) FROM violations GROUP BY day ORDER BY day ASC")
    timeline = cursor.fetchall()
    
    conn.close()
    
    return {
        "total_violations": total_violations,
        "junctions": junctions,
        "vehicles": vehicles,
        "violation_types": list(violation_counts.items()),
        "timeline": timeline
    }

def clear_db(db_path=DEFAULT_DB_PATH):
    """Helper to clear database logs for testing."""
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM violations")
        conn.commit()
        conn.close()
