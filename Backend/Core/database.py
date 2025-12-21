import sqlite3

def init_db(db_path="video_search.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Table for videos
    cursor.execute('''CREATE TABLE IF NOT EXISTS videos 
                     (video_id INTEGER PRIMARY KEY, path TEXT UNIQUE)''')
    # Table for frames/vectors
    cursor.execute(''' CREATE TABLE IF NOT EXISTS frames 
                    (   vector_id INTEGER PRIMARY KEY, 
                        video_id INTEGER, 
                        timestamp REAL, filename TEXT,
                        FOREIGN KEY (video_id) REFERENCES videos (video_id)
                    )
                   ''')
    conn.commit()
    return conn

def get_or_insert_video(conn, video_path):
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO videos (path) VALUES (?)", (video_path,))
    conn.commit()
    cursor.execute("SELECT video_id FROM videos WHERE path = ?", (video_path,))
    return cursor.fetchone()[0]