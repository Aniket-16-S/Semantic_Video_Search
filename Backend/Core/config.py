import logging
import os

logging.basicConfig ( 
    level=logging.DEBUG,  
    format='%(name)s - %(levelname)s - %(message)s'
    ) 
current_dir = os.path.dirname(os.path.abspath(__file__))

# @ video_processor.py :

FFMPEG = r'C:\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe'
INPUT_FOLDER = os.path.join(current_dir, "Backend", "Input")
DATA_FOLDER = os.path.join(current_dir, "Backend", "Data")
VP_Thread = 2


# @ SigLip_engine.py :

DATA_FOLDER = "./Backend/Data"
INDEX_FILE = "vector_storage.index"
METADATA_FILE = "metadata.json"

MODEL_NAME = "google/siglip-base-patch16-224"

BATCH_SIZE = 32  
NUM_WORKERS = 4  

