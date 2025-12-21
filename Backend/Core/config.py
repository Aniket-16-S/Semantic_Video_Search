import logging

logging.basicConfig ( 
    level=logging.DEBUG,  
    format='%(name)s - %(levelname)s - %(message)s'
    ) 


# @ video_processor.py :

FFMPEG = r'C:\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe'
input_folder = "./Backend/input"
output_folder = "./Backend/Data"
VP_Thread = 2


# @ SigLip_engine.py :

DATA_FOLDER = "./Backend/Data"
INDEX_FILE = "vector_storage.index"
METADATA_FILE = "metadata.json"

MODEL_NAME = "google/siglip-base-patch16-224"

BATCH_SIZE = 32  
NUM_WORKERS = 4  

