"""
Tag History backend — FastAPI + (Postgres in production / SQLite for local dev).

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

PERSISTENCE:
  If the DATABASE_URL environment variable is set (e.g. a free Neon/Supabase/Render
  Postgres connection string), this app stores data there — which survives restarts,
  redeploys, and free-tier sleep/wake cycles.
  If DATABASE_URL is NOT set, it falls back to a local SQLite file (tag_history.db)
  for quick local testing — but on Render's free tier that file gets wiped whenever
  the service restarts, so DATABASE_URL should always be set in production.
"""

import os
import io
import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DB_PATH = os.environ.get("TAG_DB_PATH", "tag_history.db")
ADMIN_PASSCODE = os.environ.get("ADMIN_PASSCODE", "Yulu@2026")

USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

app = FastAPI(title="Tag History API")


# ---------- database abstraction: same call sites work for both backends ----------
def get_conn():
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def ph(n):
    """Placeholder string for n params, matching the active DB's paramstyle."""
    mark = "%s" if USE_POSTGRES else "?"
    return ", ".join([mark] * n)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tag_events (
                id SERIAL PRIMARY KEY,
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tag_number ON tag_events(tag_number)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_center ON tag_events(center)")
    else:
        cur.execute("""
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tag_number ON tag_events(tag_number)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_center ON tag_events(center)")
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

    del work
    del df

    conn = get_conn()
    cur = conn.cursor()

    BATCH_SIZE = 20000
    total_inserted = 0

    if USE_POSTGRES:
        insert_sql = """
            INSERT INTO tag_events
            (tag_number, timestamp, date, time, state, center, city, technician, sequence, serial, issues)
            VALUES %s
            ON CONFLICT (tag_number, timestamp, state, center, technician, issues, sequence, serial)
            DO NOTHING
        """
        cur.execute("SELECT COUNT(*) FROM tag_events")
        before_count = cur.fetchone()[0]
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            psycopg2.extras.execute_values(cur, insert_sql, batch, page_size=BATCH_SIZE)
            conn.commit()
        cur.execute("SELECT COUNT(*) FROM tag_events")
        after_count = cur.fetchone()[0]
        total_inserted = after_count - before_count
    else:
        insert_sql = f"""
            INSERT OR IGNORE INTO tag_events
            (tag_number, timestamp, date, time, state, center, city, technician, sequence, serial, issues)
            VALUES ({ph(11)})
        """
        before = conn.total_changes
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            conn.executemany(insert_sql, batch)
            conn.commit()
        total_inserted = conn.total_changes - before

    skipped = len(records) - total_inserted
    conn.close()
    return total_inserted, skipped


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
            reader = pd.read_csv(io.BytesIO(content), chunksize=50000)
            del content
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


def rows_as_dicts(cur, rows):
    if USE_POSTGRES:
        colnames = [desc[0] for desc in cur.description]
        return [dict(zip(colnames, r)) for r in rows]
    else:
        return [dict(r) for r in rows]


@app.get("/api/tag/{tag_number}")
def get_tag_history(tag_number: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"SELECT date, time, state, center, city, technician, sequence, serial, issues, timestamp "
        f"FROM tag_events WHERE tag_number = {ph(1)} ORDER BY timestamp ASC",
        (tag_number,)
    )
    rows = cur.fetchall()
    result = rows_as_dicts(cur, rows)
    conn.close()
    if not result:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"tag": tag_number, "events": result}


@app.get("/api/search")
def search_tags(prefix: str = "", center: Optional[str] = None, limit: int = 30):
    conn = get_conn()
    cur = conn.cursor()
    like_pattern = f"%{prefix.upper()}%"
    if center:
        query = f"""SELECT DISTINCT tag_number FROM tag_events
                   WHERE tag_number LIKE {ph(1)} AND tag_number IN
                   (SELECT tag_number FROM tag_events WHERE center = {ph(1)})"""
        params = [like_pattern, center]
    else:
        query = f"SELECT DISTINCT tag_number FROM tag_events WHERE tag_number LIKE {ph(1)}"
        params = [like_pattern]
    query += f" ORDER BY tag_number LIMIT {ph(1)}"
    params.append(limit)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return {"tags": [r[0] for r in rows]}


@app.get("/api/centers")
def list_centers():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT center FROM tag_events WHERE center IS NOT NULL AND center != '' ORDER BY center"
    )
    rows = cur.fetchall()
    conn.close()
    return {"centers": [r[0] for r in rows]}


@app.get("/api/stats")
def get_stats():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT tag_number) FROM tag_events")
    total_tags = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tag_events")
    total_events = cur.fetchone()[0]
    conn.close()
    return {
        "total_tags": total_tags,
        "total_events": total_events,
        "storage_backend": "postgres" if USE_POSTGRES else "sqlite (local, NOT persistent on Render free tier)"
    }


# ---------- static frontend ----------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.get("/admin")
def serve_admin():
    return FileResponse("static/admin.html")
