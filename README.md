# Photon
IE 643 Project

# 2D dance movement generation

This repository contains the implementation of a **2D Dance Move Generation Model**. The model uses **OpenPose** for pose extraction and **Librosa** for audio analysis to generate 2D dance moves based on audio inputs. A simple **GUI interface** is provided for user interaction.

---

## Features

- Generate 2D dance moves from audio input files.
- Pre-trained model included for generating high-quality dance moves.
- Sample `.wav` files included to get started quickly.
- User-friendly GUI for interaction and visualization.

---

## Getting Started

Follow the steps below to set up and run the GUI interface:

### Prerequisites

1. Ensure Python 3.7+ is installed on your system.
2. Install the required dependencies by running:  
   ```bash
   pip install -r requirements.txt
File Structure
Learning2Dance_CAG_2020/app.py: Main script to launch the GUI interface.
samples/: Folder containing four sample .wav files for testing.
model/: Pre-trained model files (automatically loaded when running the GUI).
Running the GUI
Navigate to the repository directory:

bash
Copy code
cd Learning2Dance_CAG_2020
Run the app.py file to launch the GUI interface:

bash
Copy code
python app.py
Interact with the GUI:

Use the interface to select one of the sample .wav files or upload your own.
The generated 2D dance moves will be displayed after processing.
Example Files
The repository includes four sample .wav files located in the samples/ folder. These can be used to test the functionality of the model. Simply select a file from the GUI and watch the generated dance moves.
