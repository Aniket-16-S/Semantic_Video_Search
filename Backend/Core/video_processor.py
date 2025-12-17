import shutil
import subprocess
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import time

total = 0.0
iterations = 4

def extract_frames(video_path, output_root, method='fast', use_gpu=True):
    """
    method='fast': Extracts I-frames (Keyframes) only. Ultra fast.
    method='accurate': Uses Scene Detect + Time Interval.
    use_gpu=True: Uses NVIDIA hardware acceleration (requires NVDEC-capable GPU).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    ffmpeg_path = r'C:\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe'

    if method == 'fast':
        input_args = []
        if use_gpu:
            # NVIDIA GPU Decoding (much faster)
            # -hwaccel cuda: Use CUDA for decoding
            # -hwaccel_output_format cuda: Keep frames in GPU memory (optional, often faster to keep)
            # But for saving images, we need to download from GPU, so just -hwaccel cuda is usually enough
            
            # input_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'] 
            pass
        
        command = [
            ffmpeg_path,
            *input_args,
            '-skip_frame', 'nokey',  # Key speed hack: only process I-frames
            '-i', video_path,
            '-vsync', '0',           # '0' or 'passthrough' prevents duplicating frames
            '-frame_pts', '1',
            '-q:v', '2',             # Quality (2-31, 2 is best)
            '-loglevel', 'error',
            '-f', 'image2',          # Explicit force image2 muxer
            os.path.join(output_dir, 'frame_%d.jpg')
        ]
   
    else: # method == 'accurate'
        filter_expr = "select='isnan(prev_selected_t)+gt(scene,0.4)+gt(t-prev_selected_t,2)'"
        
        command = [
            ffmpeg_path,
            '-i', video_path,
            '-vf', filter_expr,
            '-vsync', 'vfr',
            '-frame_pts', '1',
            '-q:v', '2',
            '-loglevel', 'error',
            os.path.join(output_dir, 'frame_%d.jpg')
        ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"Done: {video_name}"
    except subprocess.CalledProcessError as e:
        return f"Error {video_name}: {e.stderr.decode()}"

if __name__ == "__main__":
    input_folder = "./Backend/input"
    output_folder = "./Backend/Data"
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        exit()

    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    


    total = 0.0
    for i in range(6) :

        print(f"Processing {len(video_files)} videos...")
        
        # OPTIMIZATION: Don't use 32 workers unless you have 32 PHYSICAL cores.
        # IO bottlenecks will slow you down. 
        # Recommended: 4 to 8 for HDD/SATA SSD, 8-12 for NVMe.
        # optimal_workers = min(os.cpu_count(), 8) 
        optimal_workers = 4
        """
        Good Performace at Cold Start
        """
        t2 = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Toggle use_gpu=True if you have an NVIDIA card
            results = executor.map(lambda v: extract_frames(v, output_folder, method='fast', use_gpu=True), video_files)
            
        t1 = time.perf_counter()
        
        # Force evaluation of the generator to print results and ensure completion
        results_list = list(results)
        total += t1-t2
        print(f"Time taken: {t1-t2}, gpu=True, at {optimal_workers}")
        # for res in results_list: print(res) # Optional: print only if debugging
        shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
    
    total /= 6
    print(f"\n\nFinal time = {total} with GPU at 4")

    

    # total = 0.0
    # for i in range(6) :

    #     print(f"Processing {len(video_files)} videos...")
        
    #     # OPTIMIZATION: Don't use 32 workers unless you have 32 PHYSICAL cores.
    #     # IO bottlenecks will slow you down. 
    #     # Recommended: 4 to 8 for HDD/SATA SSD, 8-12 for NVMe.
    #     # optimal_workers = min(os.cpu_count(), 8) 
    #     optimal_workers = 32
    #     t2 = time.perf_counter()
        
    #     with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
    #         # Toggle use_gpu=True if you have an NVIDIA card
    #         results = executor.map(lambda v: extract_frames(v, output_folder, method='fast', use_gpu=True), video_files)
            
    #     t1 = time.perf_counter()
        
    #     # Force evaluation of the generator to print results and ensure completion
    #     results_list = list(results)
    #     total += t1-t2
    #     print(f"Time taken: {t1-t2}, gpu=True, at {optimal_workers}")
    #     # for res in results_list: print(res) # Optional: print only if debugging
    #     shutil.rmtree(output_folder)
    #     os.makedirs(output_folder, exist_ok=True)
    
    # total /= 4
    # print(f"\n\nFinal time = {total} with GPU at 32")

