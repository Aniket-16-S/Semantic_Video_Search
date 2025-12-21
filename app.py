import time
tx1 = time.perf_counter()
from Backend.Core import config
import os
import sys
import logging
# Ensure 'Core' is in the path so we can import modules from it
current_dir = os.path.dirname(os.path.abspath(__file__))
# core_path = os.path.join(current_dir, 'Core')
# sys.path.append(core_path)
from Backend.Core.video_processor import bulk_extract_frames
from Backend.Core.SigLip_engine import process_and_index
from Backend.Core.search_pipeline import search_with_temporal_filter # Note: Typo in original file name 'piplline'


log = logging.getLogger(__name__)

# Constants (Could be moved to config)
INPUT_FOLDER = os.path.join(current_dir, "Backend")
INPUT_FOLDER = os.path.join(INPUT_FOLDER, "Input")
DATA_FOLDER = os.path.join(current_dir, "Backend")
DATA_FOLDER = os.path.join(DATA_FOLDER, "Data")
DB_PATH = os.path.join(current_dir, "video_search.db")

tx2 = time.perf_counter()

log.debug(f"Time for setup = {tx2-tx1}")

def run_ingestion():
    """
    Runs Backend/core/Video_processor.py and Backend/core/SigLip_engine.py
    """
    log.info("--- Starting Pipeline ---")
    
    
    log.info(f"Extracting frames from {INPUT_FOLDER} to {DATA_FOLDER}...")
    
    t1 = time.perf_counter()

    complete = bulk_extract_frames(
        input_folder=INPUT_FOLDER, 
        output_folder=DATA_FOLDER, 
        use_gpu=True 
    )
    
    if not complete :
        log.error("Error while extracting frames.")
        exit(1)
    
    log.info("Processing and Indexing images...")
    
    t2 = time.perf_counter()
    
    process_and_index(DATA_FOLDER) 
    
    t3 = time.perf_counter()
    
    log.info("--- Ingestion Complete ---")
    log.debug(f"Time for Frame Extraction = {t2-t1}\nTime for AI proccessing = {t3-t2}\nTotal = {t3-t1}")

def run_search(query):
    """
    Step 3: Perform search.
    """
    log.info(f"Searching for: '{query}'")
    t1 = time.perf_counter()
    results = search_with_temporal_filter(query, k=5)
    t2 = time.perf_counter()
    log.debug(f"Time for succesfull Search = {t2-t1}\n")
    print(f"\n--- Search Results --- (for {query=})")
    for idx, res in enumerate(results):
        print(f"{idx+1}. Time: {res['timestamp']}s | Score: {res['score']:.4f} | File: {res['filename']}")
    print("----------------------\n")

if __name__ == "__main__":
    while True :

        command = 'search'
        if command == "ingest":
            run_ingestion()
        elif command == "search":
            run_search(input("Enter query : "))
        else:
            break
