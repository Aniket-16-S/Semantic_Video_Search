import sqlite3

def init_db(db_path="video_search.db"):
    """
    Initialize the database with the given path.
    
    Args:
        db_path (str, optional): Path to the database file. Defaults to "video_search.db".
    """
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
    """
    Get or insert a video into the database.
    
    Args:
        conn (sqlite3.Connection): Database connection.
        video_path (str): Path to the video file.
    
    Returns:
        int: Video ID.
    """
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO videos (path) VALUES (?)", (video_path,))
    conn.commit()
    cursor.execute("SELECT video_id FROM videos WHERE path = ?", (video_path,))
    return cursor.fetchone()[0]

def get_max_vector_id(conn):
    """
    Get the maximum vector ID from the database.
    
    Args:
        conn (sqlite3.Connection): Database connection.
    
    Returns:
        int: Maximum vector ID.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(vector_id) FROM frames")
    res = cursor.fetchone()[0]
    return res if res is not None else -1

def get_vector_ids(conn, video_id):
    """
    Get the vector IDs for a given video ID.
    
    Args:
        conn (sqlite3.Connection): Database connection.
        video_id (int): Video ID.
    
    Returns:
        list: List of vector IDs.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT vector_id FROM frames WHERE video_id = ?", (video_id,))
    return [row[0] for row in cursor.fetchall()]

def delete_video_data(conn, video_id):
    """
    Delete video data from the database.
    
    Args:
        conn (sqlite3.Connection): Database connection.
        video_id (int): Video ID.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
    cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
    conn.commit()

def get_existing_filenames(conn):
    """
    Get the existing filenames from the database.
    
    Args:
        conn (sqlite3.Connection): Database connection.
    
    Returns:
        set: Set of existing filenames.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM frames")
    return set(row[0] for row in cursor.fetchall())

def get_video_id_from_path(conn, path):
    """
    Get the video ID from the database.
    
    Args:
        conn (sqlite3.Connection): Database connection.
        path (str): Path to the video file.
    
    Returns:
        int: Video ID.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT video_id FROM videos WHERE path = ?", (path,))
    res = cursor.fetchone()
    return res[0] if res else None

