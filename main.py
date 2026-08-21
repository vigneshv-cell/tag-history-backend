"""
Tag History backend — FastAPI + SQLite.

Two pages:
  /            -> team search page (type a tag number, see full history)
  /admin       -> upload page (passcode-protected) for loading new data

API:
  GET  /api/tag/{tag_number}       -> full history for one tag
  GET  /api/search?prefix=Y008&limit=30&center=Recircle+BLR
                                    -> matching tag numbers for autocomplete
  GET  /api/centers                -> distinct list of centers for the filter dropdown
  GET  /api/stats                  -> total tags / events / last updated
  POST /api/upload                 -> admin file upload (passcode required)
"""

import os
import io
import sqlite3
import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

DB_PATH = os.environ.get("TAG_DB_PATH", "tag_history.db")
ADMIN_PASSCODE = os.environ.get("ADMIN_PASSCODE", "Yulu@2026")  # override via Render env var for real deployments

app = FastAPI(title="Tag History API")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tag_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_number TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            state TEXT,
            center TEXT,
            city TEXT,
            technician TEXT,
            sequence INTEGER,
            serial TEXT,
            issues TEXT,
            UNIQUE(tag_number, timestamp, state, center, technician, issues, sequence, serial)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_number ON tag_events(tag_number)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_center ON tag_events(center)")
    conn.commit()
    conn.close()


init_db()


# ---------- column matching, mirrors the earlier client-side logic ----------
def find_col(columns, candidates):
    lower_map = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        for lc, orig in lower_map.items():
            if cand.lower() in lc:
                return orig
    return None


def parse_timestamp(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        # Excel serial date
        try:
            base = datetime.datetime(1899, 12, 30)
            return base + datetime.timedelta(days=float(value))
        except Exception:
            return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def ingest_dataframe(df: pd.DataFrame):
    columns = list(df.columns)
    tag_col = find_col(columns, ["tag_number", "tag number", "tag"])
    state_col = find_col(columns, ["state", "status"])
    center_col = find_col(columns, ["yz_name", "center", "refurb center"])
    city_col = find_col(columns, ["city"])
    tech_col = find_col(columns, ["first_name", "technician", "created_by_name"])
    seq_col = find_col(columns, ["tag_sequence", "sequence"])
    issues_col = find_col(columns, ["issue_names", "issues"])
    serial_col = find_col(columns, ["serial_number", "serial number", "serial"])
    ts_col = find_col(columns, ["created_dt_timestamp_ist", "timestamp", "created_dt_timestamp"])
    date_col = find_col(columns, ["_date", "date"])

    if not tag_col:
        raise ValueError('Could not find a "tag_number" column in this file.')

    work = pd.DataFrame(index=df.index)
    work['tag_number'] = df[tag_col].astype(str).str.strip()

    if ts_col:
        dt = pd.to_datetime(df[ts_col], errors='coerce')
    else:
        dt = pd.Series(pd.NaT, index=df.index)
    if date_col:
        fallback = pd.to_datetime(df[date_col], errors='coerce')
        dt = dt.fillna(fallback)
    work['dt'] = dt

    def clean_str_col(col, default=''):
        if col:
            s = df[col].astype(str).str.strip()
            s = s.replace('nan', default)
            return s
        return pd.Series(default, index=df.index)

    work['state'] = clean_str_col(state_col)
    work['center'] = clean_str_col(center_col)
    work['city'] = clean_str_col(city_col)
    work['technician'] = clean_str_col(tech_col).replace('', 'Unknown')

    if seq_col:
        work['sequence'] = pd.to_numeric(df[seq_col], errors='coerce').fillna(1).astype(int)
    else:
        work['sequence'] = 1

    work['serial'] = clean_str_col(serial_col)
    work['issues'] = clean_str_col(issues_col)

    # drop rows with no usable timestamp or tag number
    work = work[work['dt'].notna()]
    work = work[work['tag_number'].str.len() > 0]
    work = work[work['tag_number'].str.lower() != 'nan']

    work['date'] = work['dt'].dt.strftime('%Y-%m-%d')
    work['time'] = work['dt'].dt.strftime('%H:%M:%S')
    work['timestamp'] = work['date'] + ' ' + work['time']

    records = list(work[[
        'tag_number', 'timestamp', 'date', 'time', 'state', 'center',
        'city', 'technician', 'sequence', 'serial', 'issues'
    ]].itertuples(index=False, name=None))

    # free the dataframes before the DB step to reduce peak memory
    del work
    del df

    conn = get_conn()
    before = conn.total_changes

    # insert in batches instead of one giant executemany, and commit incrementally
    BATCH_SIZE = 20000
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        conn.executemany(
            """INSERT OR IGNORE INTO tag_events
               (tag_number, timestamp, date, time, state, center, city, technician, sequence, serial, issues)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch
        )
        conn.commit()

    inserted = conn.total_changes - before
    skipped = len(records) - inserted
    conn.close()
    return inserted, skipped


@app.post("/api/upload")
async def upload_file(passcode: str = Form(...), file: UploadFile = File(...)):
    if passcode != ADMIN_PASSCODE:
        raise HTTPException(status_code=401, detail="Incorrect passcode")

    content = await file.read()
    filename = file.filename or ""

    total_rows = 0
    total_inserted = 0
    total_skipped = 0

    try:
        if filename.lower().endswith(".csv"):
            # Stream the CSV in chunks so we never hold the whole file in memory at once —
            # important on memory-constrained free hosting tiers.
            reader = pd.read_csv(io.BytesIO(content), chunksize=50000)
            del content  # free the raw bytes now that pandas has its own buffer
            for chunk_df in reader:
                total_rows += len(chunk_df)
                inserted, skipped = ingest_dataframe(chunk_df)
                total_inserted += inserted
                total_skipped += skipped
        else:
            df = pd.read_excel(io.BytesIO(content))
            del content
            total_rows = len(df)
            total_inserted, total_skipped = ingest_dataframe(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    return {
        "filename": filename,
        "rows_in_file": total_rows,
        "inserted": total_inserted,
        "skipped_duplicates": total_skipped
    }


@app.get("/api/tag/{tag_number}")
def get_tag_history(tag_number: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT date, time, state, center, city, technician, sequence, serial, issues, timestamp "
        "FROM tag_events WHERE tag_number = ? ORDER BY timestamp ASC",
        (tag_number,)
    ).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"tag": tag_number, "events": [dict(r) for r in rows]}


@app.get("/api/search")
def search_tags(prefix: str = "", center: Optional[str] = None, limit: int = 30):
    conn = get_conn()
    query = "SELECT DISTINCT tag_number FROM tag_events WHERE tag_number LIKE ?"
    params = [f"%{prefix.upper()}%"]
    if center:
        query = """SELECT DISTINCT tag_number FROM tag_events
                   WHERE tag_number LIKE ? AND tag_number IN
                   (SELECT tag_number FROM tag_events WHERE center = ?)"""
        params = [f"%{prefix.upper()}%", center]
    query += " ORDER BY tag_number LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return {"tags": [r["tag_number"] for r in rows]}


@app.get("/api/centers")
def list_centers():
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT center FROM tag_events WHERE center IS NOT NULL AND center != '' ORDER BY center"
    ).fetchall()
    conn.close()
    return {"centers": [r["center"] for r in rows]}


@app.get("/api/stats")
def get_stats():
    conn = get_conn()
    total_tags = conn.execute("SELECT COUNT(DISTINCT tag_number) AS c FROM tag_events").fetchone()["c"]
    total_events = conn.execute("SELECT COUNT(*) AS c FROM tag_events").fetchone()["c"]
    conn.close()
    return {"total_tags": total_tags, "total_events": total_events}


# ---------- static frontend ----------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.get("/admin")
def serve_admin():
    return FileResponse("static/admin.html")
