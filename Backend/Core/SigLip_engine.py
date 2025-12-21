import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipVisionModel, SiglipProcessor
from PIL import Image
import faiss
from tqdm import tqdm
from Backend.Core.database import init_db, get_or_insert_video
from Backend.Core.config import *

log = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, folder_path, processor):
        self.folder_path = folder_path
        self.processor = processor
        self.image_paths = []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conn = init_db()
    
    # Load Model
    model = SiglipVisionModel.from_pretrained(MODEL_NAME).to(device)
    processor = SiglipProcessor.from_pretrained(MODEL_NAME)
    
    dataset = ImageDataset(data_folder, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)
    
    index = faiss.IndexFlatIP(768)
    
    log.info(f"Starting Embedding with SigLIP on {device}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None : 
                continue
            
            pixel_values, paths = batch
            outputs = model(pixel_values=pixel_values.to(device))
            embeddings = F.normalize(outputs.pooler_output, p=2, dim=1)
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            
            start_id = index.ntotal
            index.add(embeddings_np)
            
            cursor = conn.cursor()
            for i, path in enumerate(paths):
                # Extract timestamp from filename (assuming FFmpeg -frame_pts 1)
                # Filename format: videoName_frame_123.jpg where 123 is the PTS
                filename = os.path.basename(path)
                try:
                    main = filename.split('_')
                    pts = float( main[-1].replace('.jpg', '').replace('pts=', '') )
                    fps = float( main[1].replace('fps=', '') )
                    # If 24fps, timestamp = pts / 24. For simplicity, we store PTS.
                    timestamp = pts / fps
                except:
                    timestamp = 0.0

                video_id = get_or_insert_video(conn, path) # Simplified for example
                
                cursor.execute("""INSERT INTO frames (vector_id, video_id, timestamp, filename) 
                                  VALUES (?, ?, ?, ?)""", 
                               (start_id + i, video_id, timestamp, filename))
            conn.commit()

    faiss.write_index(index, "vector_storage.index")
    log.info("Indexing Complete.\n SQL Database Updated.")



if __name__ == '__main__' :
    process_and_index(DATA_FOLDER) 