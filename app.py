import time
import os
from tkinter import filedialog
from Backend.Core.video_processor import bulk_extract_frames
from Backend.Core.SigLip_engine import process_and_index, remove_video_vectors
from Backend.Core.search_pipeline import search_with_temporal_filter 
from Backend.Core.config import *

log = logging.getLogger(__name__)


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

def add_videos_flow():
    
    choice = input("Select Source - Do you want to add a whole FOLDER? (y/n)\n : ")
    
    video_files = []
    
    if choice.lower() == 'y':
        folder = filedialog.askdirectory(title="Select Video Folder")
        if not folder:
            return
        import glob
        video_files = glob.glob(os.path.join(folder, "*.mp4"))
        if not video_files:
            print("Error", "No .mp4 files found in that folder.")
            return
    elif choice.lower() == 'n' : 
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("MP4 Videos", "*.mp4")])
        if not files: return
        video_files = list(files)
    else :
        print("invalod input")
        return

    typs = ['fast', 'accurate', '1fps']
    method = input("\nExtraction Method : \n1.fast \n2.accurate \n3.1fps \n : ")
    try :
        method = typs[int(method)-1]
        
    except Exception :
        print("Invalid Chice")
        return
        
    
    print(f"Selected {len(video_files)} videos. Method: {method}. Processing...")
    
    success = bulk_extract_frames(
        output_folder=DATA_FOLDER, 
        use_gpu=True,
        method=method,
        video_files_list=video_files
    )
    
    if not success:
        print("Error during frame extraction.")
        return

    print("Indexing...")
    process_and_index(DATA_FOLDER)
    return
    

def remove_videos_flow():
    files = filedialog.askopenfilenames(title="Select Videos to Remove", filetypes=[("MP4 Videos", "*.mp4")])
    if not files: 
        return
    
    for vid in files:
        print(f"Removing {vid}...")
        remove_video_vectors(vid)
        
    print("Success", "Selected videos removed from index.")

def main():
    print("AI Video Search Engine Started.")

    while True:        
        choice = input("\n1.Search\n2.Add Videos\n3.Remove videos\n4.Exit\n : ")
        try :
            choice = int(choice)
        except :
            continue
        if choice == 1 :
             query = input("Enter search query : ")
             if query:
                run_search(query)
                input("\nPress Enter to continue...")
                 
        elif choice == 2:
            add_videos_flow()
            
        elif choice == 3:
            remove_videos_flow()
            
        elif choice == 4 or choice == "":
            break
            
    print("Exited\n")

if __name__ == "__main__":
    main()
