import logging

ffmpeg_path = r'C:\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe'

logging.basicConfig ( 
    level=logging.DEBUG,  
    format='%(name)s - %(levelname)s - %(message)s', 
    filemode='a' 
    ) 

# # Log messages
# logging.debug("This is a debug message")
# logging.info("This is an info message")
# logging.warning("This is a warning message")
# logging.error("This is an error message")
# logging.critical("This is a critical message")
