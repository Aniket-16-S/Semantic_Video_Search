import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipVisionModel, SiglipProcessor
from PIL import Image
import faiss
from tqdm import tqdm
import numpy as np
import shutil
from Backend.Core.database import init_db, get_or_insert_video, get_max_vector_id, get_existing_filenames, get_vector_ids, delete_video_data, get_video_id_from_path
from Backend.Core.config import *

log = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    Dataset for loading and processing images from a folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
        processor (SiglipProcessor): SigLIP processor for image processing.
        exclude_files (list, optional): List of filenames to exclude from processing.
    """
    def __init__(self, folder_path, processor, exclude_files=None):
        self.folder_path = folder_path
        self.processor = processor
        self.image_paths = []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        # Filtering out already indexed files
        if exclude_files:
             # Filenames in DB are basenames
             self.image_paths = [p for p in all_files if os.path.basename(p) not in exclude_files]
        else:
             self.image_paths = all_files
        
        # Sort paths so the index order is deterministic
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            # return pixel_values directly. SigLIP processor returns a dict.
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "path": img_path,
                "valid": True
            }
        except Exception as e:
            log.error(f"Error loading {img_path}: {e}")
            return {"valid": False}

def custom_collate_fn(batch):
    batch = [item for item in batch if item["valid"]]
    if not batch: return None
    
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    paths = [item["path"] for item in batch]
    return pixel_values, paths

def process_and_index(data_folder):
    """
    Process and index images in the given folder.
    
    Args:
        data_folder (str): Path to the folder containing images to be indexed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conn = init_db()
    
    model = SiglipVisionModel.from_pretrained(MODEL_NAME).to(device)
    processor = SiglipProcessor.from_pretrained(MODEL_NAME)
    
    existing_files = get_existing_filenames(conn)
    dataset = ImageDataset(data_folder, processor, exclude_files=existing_files)
    
    if len(dataset) == 0:
        log.info("No new images to index.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)
    
    
    if os.path.exists("vector_storage.index"):
        try:
            index = faiss.read_index("vector_storage.index")
        except:
            log.warning("Could not read existing index, creating new IndexIDMap.")
            sub_index = faiss.IndexFlatIP(768)
            index = faiss.IndexIDMap(sub_index)
    else:
        sub_index = faiss.IndexFlatIP(768)
        index = faiss.IndexIDMap(sub_index)
    
    log.info(f"Starting Embedding with SigLIP on {device}")
    
    current_max_id = get_max_vector_id(conn)
    start_id = current_max_id + 1

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None : 
                continue
            
            pixel_values, paths = batch
            outputs = model(pixel_values=pixel_values.to(device))
            embeddings = F.normalize(outputs.pooler_output, p=2, dim=1)
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            
            batch_size = embeddings_np.shape[0]
            ids = np.arange(start_id, start_id + batch_size).astype('int64')
            
            index.add_with_ids(embeddings_np, ids)
            
            cursor = conn.cursor()
            for i, path in enumerate(paths):
                filename = os.path.basename(path)
                try:
                    main = filename.split('_')
    
                    pts_str = [s for s in main if 'pts=' in s][0]
                    fps_str = [s for s in main if 'fps=' in s][0]

                    pts = float( pts_str.replace('.jpg', '').replace('pts=', '') )
                    fps = float( fps_str.replace('fps=', '') )
                    timestamp = pts * fps
                except:
                    timestamp = 0.0
                vid_row_id = get_or_insert_video(conn, path) 
                
                cursor.execute("""INSERT INTO frames (vector_id, video_id, timestamp, filename) 
                                  VALUES (?, ?, ?, ?)""", 
                               (int(ids[i]), vid_row_id, timestamp, filename))
            
            conn.commit()
            start_id += batch_size

    faiss.write_index(index, "vector_storage.index")
    log.info("Indexing Complete.\n SQL Database Updated.")


def remove_video_vectors(video_path):
    """
    Removes a video and its vectors. 
    Problem: 'video_path' input in Add mode is the video file. 
    But in DB 'videos' table, we are storing FRAME paths (based on previous analysis).
    Wait, let's check get_or_insert_video usage in original code.
    Original: video_id = get_or_insert_video(conn, path) where path is the image path.
    This means 'videos' table actually stored image paths, which is wrong design for "Video" search.
    However, to support "Remove Video", we face a challenge: We need to find all frames belonging to a video.
    Frames are named "VideoName_fps=...".
    So we can query by filename LIKE 'VideoName_%'.
    """
    conn = init_db()
    video_basename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]
    
    cursor = conn.cursor()
    query_pattern = f"{video_name_no_ext}_fps=%"
    cursor.execute("SELECT vector_id, video_id, filename FROM frames WHERE filename LIKE ?", (query_pattern,))
    rows = cursor.fetchall()
    
    if not rows:
        log.warning(f"No frames found for video: {video_name_no_ext}")
        return

    vector_ids = [r[0] for r in rows]
    video_ids = set(r[1] for r in rows)
    
    if os.path.exists("vector_storage.index"):
        index = faiss.read_index("vector_storage.index")
        ids_to_remove = np.array(vector_ids).astype('int64')
        index.remove_ids(ids_to_remove)
        faiss.write_index(index, "vector_storage.index")
    
    cursor.execute("DELETE FROM frames WHERE filename LIKE ?", (query_pattern,))
    for vid in video_ids:
        cursor.execute("DELETE FROM videos WHERE video_id = ?", (vid,))
    
    conn.commit()
    log.info(f"Removed {len(vector_ids)} vectors for video: {video_name_no_ext}")


if __name__ == '__main__' :
    process_and_index(DATA_FOLDER)
 