from tqdm import tqdm
import argparse
import os

def get_videos_path(dataset_path):
    # Add debug prints
    print("Searching in directory:", dataset_path)
    videos_path = []
    for path, subdirs, files in os.walk(dataset_path):
        print("Current path:", path)
        print("Subdirs found:", subdirs)
        print("Files found:", files)
        for name in files:
            if name.endswith('.mp4'):
                full_path = os.path.join(path, name)
                videos_path.append(full_path)
    print("Found video paths:", videos_path)
    return videos_path

def extract_audio(video_path):
    index = video_path.rfind('I')   
    full_video_name = video_path[index:]
    video_name = video_path[:index] + full_video_name.split('.mp4')[0]

    # Modified for Windows
    cmd = f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec copy "{video_name}.aac"'
    print("Executing command:", cmd)
    os.system(cmd)

    cmd = f'ffmpeg -loglevel error -i "{video_name}.aac" "{video_name}.wav"'
    os.system(cmd)

    os.system(f'del "{video_name}.aac"')

def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio data from videos in .wav format.")
    parser.add_argument('--dataset_path', default="", help='Path to root of dataset to extract audios.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Dataset path:", args.dataset_path)
    
    videos_path = get_videos_path(args.dataset_path)
    print("Final video paths:", videos_path)

    for video in tqdm(videos_path, desc="Extracting audio from videos..."):
        print("Processing video:", video)
        extract_audio(video)

if __name__ == "__main__":
    main()
