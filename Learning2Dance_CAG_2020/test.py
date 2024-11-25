import torch
import numpy as np
import os
import argparse
import torchaudio
from data import *
from sigth2sound import *
from tools.utils import *

def get_audio_torch(input_path):
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] == 2:  # If stereo, convert to mono
        audio = audio.mean(axis=0).unsqueeze(0)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    audio = torchaudio.functional.mu_law_encoding(audio, 16)

    return audio.float()

def make_z_vary(idx,c,t,m):
    if idx is not None:
        np.random.seed(idx)
    else:
        np.random.seed()

    xs = np. linspace (0,1000,m) # Test input vector
    mxs = np.zeros(m) # Zero mean vector

    z = []
    for i in range(c):
        lsc = ((float(i)+1)/c)*(100*(1024/c))
        Kss = np.exp((-1*(xs[:,np.newaxis]-xs[:,np.newaxis ].T)**2)/(2*lsc**2)) # Covariance matrix
        fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular =True).rvs(1).T
        z.append(fs)
    z = np.asarray(z)
    return z

def test(args, device):
    # Load models with map_location to handle CPU/GPU
    audio_model = cnn_1d_soudnet(3)
    audio_model.load_state_dict(torch.load(args.a_ckp_path, map_location=device))
    audio_model.to(device)

    model = Generator(device, args.num_class, args.dropout, False)
    model.load_state_dict(torch.load(args.ckp_path, map_location=device))
    model.to(device)

    audio_model.eval()
    model.eval()

    # Process audio input
    audio = get_audio_torch(args.input)

    print(f"audio {audio.shape}")
    video_size = int((audio.shape[1] / 16000) * args.fps)



    # Generate the z vector using make_z_vary
    z = torch.Tensor(make_z_vary(None, 512, args.size_video, int(video_size / 16))).view(1, 512, -1, 1).to(device)
 
    print("="*20)

    k = audio[:,:int(int(audio.shape[1]/int(z.shape[2]/4))*int(z.shape[2]/4))].to(device).view(int(z.shape[2]/4),1,-1)
    print(k.shape)
    label = audio_model(audio[:,:int(int(audio.shape[1]/int(z.shape[2]/4))*int(z.shape[2]/4))].to(device).view(int(z.shape[2]/4),1,-1))
    print("="*20)
    print(label)
    label = label.argmax(1).cpu().data.to(device)
    print("="*20)
    print(label)

    draw_poses = model(label, z)

    notorch_pose = draw_poses[0].permute(1, 2, 0).cpu().data.numpy()
    try:
        os.mkdir(args.out_video)
    except:
        pass

    label_0 = label.cpu().data.tolist()[0]
    if label_0 == 0:
        video_name = '/ballet'
    elif label_0 == 1:
        video_name = '/michael'
    elif label_0 == 2:
        video_name = '/salsa'

    os.makedirs(args.out_video + video_name + '/test_img/', exist_ok=True)
    make_video(args.out_video + video_name + '/test_img/', notorch_pose, n=1500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the audio input file')
    parser.add_argument('--a_ckp_path', type=str, required=True, help='Path to audio model checkpoint')
    parser.add_argument('--ckp_path', type=str, required=True, help='Path to generator model checkpoint')
    parser.add_argument('--out_video', type=str, default='out/', help='Output video path')
    parser.add_argument('--num_class', type=int, default=3, help='Number of classes')
    parser.add_argument('--size_video', type=int, default=150, help='Number of frames in video')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output video')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')

    args = parser.parse_args()
    
    device = torch.device("cpu")  # Ensure we use CPU
    test(args, device)
