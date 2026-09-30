import subprocess
import os

def download_and_trim_youtube_audio(url, start_time, end_time, output_path):
    """
    Downloads audio directly from YouTube and trims it to the specified timestamp using ffmpeg.
    Timestamps must be in HH:MM:SS format.
    """
    temp_file = "temp_audio.wav"
    
    # Download raw audio via yt-dlp
    download_cmd = [
        "yt-dlp", "-x", "--audio-format", "wav", 
        "-o", temp_file, url
    ]
    subprocess.run(download_cmd, check=True)
    
    # Trim via ffmpeg without re-encoding
    trim_cmd = [
        "ffmpeg", "-y", "-i", temp_file, 
        "-ss", start_time, "-to", end_time, 
        "-c", "copy", output_path
    ]
    subprocess.run(trim_cmd, check=True)
    os.remove(temp_file)
    print(f"Successfully saved trimmed audio to {output_path}")

if __name__ == "__main__":
    # Example usage for extracting a lean bog event
    download_and_trim_youtube_audio(
        url="https://www.youtube.com/watch?v=example",
        start_time="00:01:14",
        end_time="00:01:22",
        output_path="../system_3_manual_recording/recordings/lean_bog_starvation/yt_sample_1.wav"
    )