import os
import subprocess

def convert_audio_files(input_folder, output_folder, target_sr=16000, target_channels=1):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the input directory
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file)

                # Run ffmpeg command to convert the audio
                ffmpeg_cmd = [
                    "ffmpeg", "-i", input_path,
                    "-ar", str(target_sr),  # Set sample rate
                    "-ac", str(target_channels),  # Set channels (1 for mono)
                    output_path
                ]
                print(f"Converting {input_path} to {output_path}...")
                subprocess.run(ffmpeg_cmd)

# Define your class directories and output directories
class_dirs = {
    "ballet": ("F:\\Sample\\audio_dataset\\Ballet", "F:\\Sample\\converted_audio\\Ballet"),
    "bangra": ("F:\\Sample\\audio_dataset\\Ballet", "F:\\Sample\\converted_audio\\bangra"),
    "salsa": ("F:\\Sample\\audio_dataset\\Ballet", "F:\\Sample\\converted_audio\\Salsa")
}

# Convert audio files for each class
for class_name, (input_dir, output_dir) in class_dirs.items():
    convert_audio_files(input_dir, output_dir)
    print(f"Finished converting files in {class_name} class.")
