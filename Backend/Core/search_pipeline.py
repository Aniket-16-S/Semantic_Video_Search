from transformers import SiglipTextModel, SiglipTokenizer
import torch
import sqlite3
import faiss
import Backend.Core.config as config
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)

def search_with_temporal_filter(query_text, k=5, time_threshold=5.0):
    """
    Perform a search with temporal filtering.
    
    Args:
        query_text (str): The text query to search for.
        k (int, optional): Number of results to return. Defaults to 5.
        time_threshold (float, optional): Time threshold for temporal filtering. Defaults to 5.0.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    index = faiss.read_index("vector_storage.index")
    conn = sqlite3.connect("video_search.db")
    
    
    inputs = tokenizer([query_text], padding="max_length", 
                       max_length=64, return_tensors="pt").to(device)
    
    with torch.no_grad():
        text_vec = model(**inputs).pooler_output
        text_vec = F.normalize(text_vec, p=2, dim=1).cpu().numpy().astype('float32')

    distances, indices = index.search(text_vec, k * 10)
    
    filtered_results = []
    seen_videos = {} # video_id -> list of timestamps already picked

    cursor = conn.cursor()
    for dist, v_id in zip(distances[0], indices[0]):
        cursor.execute("SELECT video_id, timestamp, filename FROM frames WHERE vector_id = ?", (int(v_id),))
        row = cursor.fetchone()
        if not row: continue
        
        vid_id, t_stamp, fname = row
        
        # Temporal Logic: Check if we already have a result from this video near this time
        is_too_close = False
        if vid_id in seen_videos:
            for existing_time in seen_videos[vid_id]:
                if abs(existing_time - t_stamp) < time_threshold:
                    is_too_close = True
                    break
        
        if not is_too_close:
            filtered_results.append({
                "score": float(dist),
                "timestamp": t_stamp,
                "filename": fname
            })
            
            if vid_id not in seen_videos: seen_videos[vid_id] = []
            seen_videos[vid_id].append(t_stamp)
            
        if len(filtered_results) >= k:
            break

    log.info(f"Top Result Cosine Similarity: {filtered_results[0]['score'] if filtered_results else 0}")
    return filtered_results


if __name__ == '__main__' :
    while True :
        c = input("Search for :")
        if c == 'e' :
            break
        else :
            query = c
            log.info(f"Searching for: '{query}'")
            results = search_with_temporal_filter(query, k=5)
            
            print("\n--- Search Results ---")
            for idx, res in enumerate(results):
                print(f"{idx+1}. Time: {res['timestamp']}s | Score: {res['score']:.4f} | File: {res['filename']}")
            print("----------------------\n")