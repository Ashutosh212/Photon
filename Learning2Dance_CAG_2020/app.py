import streamlit as st
import os
import torch
from pathlib import Path
import shutil
import cv2
import numpy as np

from test import get_audio_torch, make_z_vary, test

def main():
    st.title("Audio to Video Generator")

    st.sidebar.header("Settings")
    input_audio = st.sidebar.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if input_audio:
        with open("temp_audio.wav", "wb") as f:
            f.write(input_audio.getbuffer())

        st.audio("temp_audio.wav", format="audio/wav")

        # Take input parameters for the script
        num_class = st.sidebar.slider('Number of Classes', 1, 10, 3)
        size_video = st.sidebar.slider('Video Size (Frames)', 50, 200, 150)
        fps = st.sidebar.slider('Frames Per Second (FPS)', 15, 60, 30)
        dropout = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.0)

        a_ckp_path = st.sidebar.text_input("Audio Model Checkpoint Path", "E:\\IE643\\Photon\\Learning2Dance_CAG_2020\\weights\\audio_classifier.pt")
        ckp_path = st.sidebar.text_input("Generator Model Checkpoint Path", "E:\IE643\Photon\\Learning2Dance_CAG_2020\\weights\\generator.pt")
        out_video_path = "out/"  

        if st.button('Generate Video'):
            # Ensure required folders exist
            if not os.path.exists(out_video_path):
                os.makedirs(out_video_path)

            try:
                class Args:
                    def __init__(self):
                        self.input = "temp_audio.wav"
                        self.a_ckp_path = a_ckp_path
                        self.ckp_path = ckp_path
                        self.out_video = out_video_path
                        self.num_class = num_class
                        self.size_video = size_video
                        self.fps = fps
                        self.dropout = dropout

                args = Args()
                device = torch.device("cpu")
                test(args, device)

                video_dir = Path(out_video_path) / "salsa"  # Adjust based on the model's output class
                if not video_dir.exists():
                    st.error("No video found at the expected path!")
                    return

                video_files = sorted(video_dir.glob('test_img/*.mp4'))

                if not video_files:
                    st.error("No .mp4 video files found in the directory!")
                    return

                for video_file in video_files:
                    video_file = open(video_file, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)

                st.success("Videos generated and displayed successfully!")

            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()