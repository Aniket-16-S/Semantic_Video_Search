import shutil
import subprocess
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import time
from Backend.Core.config import *

log = logging.getLogger(__name__)

def extract_frames(video_path, output_root, method='fast', use_gpu=True):
    """
    Extracts frames from videos : 
        currently .mp4 -> .jpg 

    Docstring for extract_frames
        
        :param video_path: path to input video files (.mp4 only as of now)
            #TODO : add support to other encodings
        :param output_root: .jpg output root path (will create folder same as video name to place frame_names.jpg)
        
        :param method: 
        
            1. method='fast': Extracts I th - frames (Keyframes) only. hence, faster.

            2. method='accurate': Uses Scene Detect (extract frame is scene changed at given probability) 
                                    + Time Interval (if not scene changed in  x secs )
                                    Values at config .py
            
            3. method = '1fps' : extract every frame at sec.
                 
        :param use_gpu: weather to use gpu or  not (if available.)
    
        USE '-vf', "fps=1",  to Force 1 frame per second
    """
    fps_raw = subprocess.check_output([
                'ffprobe', '-v', '0', '-select_streams', 'v:0', 
                '-show_entries', 'stream=time_base', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                video_path
                ]).decode('utf-8').strip()

    # output is like -  "30/1" or "24000/1001" so converting to a clean number
    num, den = map(int, fps_raw.split('/'))
    fps = num / den
        
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = output_root
    
    if use_gpu :
        input_args = get_hw_accel_args()

    if method == 'fast':     
       
        command = [
            FFMPEG,
            *input_args,
            '-fflags', '+discardcorrupt',
            '-skip_frame', 'nokey',       #  only process key -frames
            '-i', video_path,
            '-vsync', '0',                # '0' or 'passthrough' =  prevent duplicating frames
            '-an',                        # disable audio
            '-sn',                        # disable subtitles
            '-dn',                        # disable data streams
            '-q:v', '2',                  # Quality (2-31, 2 is best)
            '-loglevel', 'error', 
            '-frame_pts', '1',

            os.path.join(output_dir, f"{video_name}_fps={fps}_pts=%08d.jpg")
        ]
   
    elif method == 'accurate' :
        
        filter_expr = "select='isnan(prev_selected_t)+gt(scene,0.4)+gt(t-prev_selected_t,2)'"
        
        command = [
            FFMPEG,
            *input_args,
            '-fflags', '+discardcorrupt',
            '-i', video_path,
            '-vf', filter_expr,
            '-vsync', 'vfr',      # as equivalent as 0 or passthrogh for images ONLY
            '-an',                # disable audio
            '-sn',                # disable subtitles
            '-dn',                # disable data streams
            '-q:v', '2',
            '-loglevel', 'error',
            '-frame_pts', '1',
            os.path.join(output_dir, f"{video_name}_fps={fps}_pts=%08d.jpg")
        ]
    
    elif method == '1fps' :

        command = [
            FFMPEG,
            *input_args,
            '-fflags', '+discardcorrupt',
            '-i', video_path,
            '-vsync', '0',
            '-vf', 'fps=1',       # 1 frame per second
            '-an',                # disable audio
            '-sn',                # disable subtitles
            '-dn',                # disable data streams
            '-q:v', '2',
            '-loglevel', 'error',
            '-frame_pts', '1',
            os.path.join(output_dir, f"{video_name}_fps={fps}_pts=%08d.jpg")
        ]


    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e :
        log.error(f"Error {video_name}: {e.stderr.decode()}")
        return str(e)


def get_hw_accel_args():
    """
    Detects the GPU vendor and returns the corresponding FFmpeg hardware 
    acceleration flags. Returns None if no GPU is detected.        

    """
    HW_ACCEL_MAP = {
        "nvidia": ["-hwaccel", "cuda"],
        "intel": ["-hwaccel", "qsv"],
        "amd": ["-hwaccel", "d3d11va"],
        "mac": ["-hwaccel", "videotoolbox"],
        "auto": ["-hwaccel", "auto"],
    }

    #TODO : Use robus logic to detect dedicated GPU Hardware.

    # system_platform = platform.system().lower()
    
    # # 1. MAC: Apple Silicon/High-end Intel Macs handle this via OS
    # if system_platform == "darwin":
    #     return HW_ACCEL_MAP["mac"]

    # # 2. NVIDIA: nvidia-smi only lists dedicated hardware
    # try:
    #     # -q gets full info, we grep for 'FB Memory' (Frame Buffer)
    #     out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL)
    #     vram = int(out.decode().strip().split('\n')[0])
    #     if vram >= 2000: # 2000 MB
    #         return HW_ACCEL_MAP["nvidia"]
    # except:
    #     pass

    # # 3. WINDOWS: Check WMI for Name and AdapterRAM
    # if system_platform == "windows":
    #     try:
    #         cmd = "wmic path win32_VideoController get Name, AdapterRAM /format:list"
    #         output = subprocess.check_output(cmd, shell=True).decode()
            
    #         # wmic output is blocks of Name=... AdapterRAM=...
    #         # We split by double newlines to check each GPU found
    #         devices = output.strip().split('\r\r\n\r\r\n')
            
    #         for device in devices:
    #             lines = device.strip().split('\n')
    #             info = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in lines if '=' in line}
                
    #             name = info.get("Name", "").lower()
    #             # AdapterRAM is in bytes. 2GB = 2147483648 bytes
    #             ram_bytes = int(info.get("AdapterRAM", 0))
                
    #             if ram_bytes >= 2 * 1024 * 1024 * 1024:
    #                 if "nvidia" in name: return HW_ACCEL_MAP["nvidia"]
    #                 if "amd" in name or "ati" in name: return HW_ACCEL_MAP["amd"]
    #                 if "arc" in name: return HW_ACCEL_MAP["intel"]
    #     except:
    #         pass

    # # 4. LINUX: Check lspci and filter by memory bars
    # elif system_platform == "linux":
    #     try:
    #         # lspci -v shows memory regions. Dedicated cards have large [size=...] bars
    #         output = subprocess.check_output("lspci -vnn | grep -A 12 'VGA'", shell=True).decode().lower()
            
    #         # Find memory sizes in the output (e.g., [size=4g])
    #         mem_sizes = re.findall(r'size=(\d+)([gmk])', output)
            
    #         has_large_vram = False
    #         for val, unit in mem_sizes:
    #             if unit == 'g' and int(val) >= 2:
    #                 has_large_vram = True
    #             elif unit == 'm' and int(val) >= 2000:
    #                 has_large_vram = True
            
    #         if has_large_vram:
    #             if "nvidia" in output: return HW_ACCEL_MAP["nvidia"]
    #             if "amd" in output or "ati" in output: return HW_ACCEL_MAP["amd"]
    #             if "arc" in output: return HW_ACCEL_MAP["intel"]
    #     except:
    #         pass

    return HW_ACCEL_MAP['auto'] # Return empty if no dedicated hardware >= 2GB found # ffmpeg will decide best or will fallback to CPU mode 


def bulk_extract_frames(input_folder=None, output_folder=None, use_gpu=True, thread_count=None, del_op_folder=False, method='fast', video_files_list=None) -> None :
    
    if video_files_list:
        video_files = video_files_list
    elif input_folder:
        if not os.path.exists(input_folder):
            log.critical(f"Error: Input folder '{input_folder}' does not exist.")
            exit()
        video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    else:
        log.error("No input folder or video list provided")
        return False
    
    os.makedirs(output_folder, exist_ok=True)
    
    if thread_count :
        optimal_workers = thread_count
    else :
        optimal_workers = 8
        # 4 or 8 preffered due to io bottle neck , (system req. to laod files in ram.)
        # 4 / 8 will give Good Performace at Cold Start
        
    t2 = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        
        results = executor.map(lambda v: extract_frames(v, output_folder, method=method, use_gpu=use_gpu), video_files)
        
    t1 = time.perf_counter()
    
    # Force evaluation of the generator to print results and ensure completion
    results_list = list(results)
    
    log.info(f"Time taken for extraction : {t1-t2}, gpu={use_gpu}, at {optimal_workers}")
    
    if del_op_folder :
        # For debigging ONLY
        shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        log.warning("Output Folder Cleared !")

    for result in results_list :
        if not result is True :
            log.error(result)
            return False
    
    return True

if __name__ == "__main__":
    log.info("Testing Frame Extractor")
    bulk_extract_frames(input_folder=INPUT_FOLDER, output_folder=DATA_FOLDER)