import torch
import numpy as np
from pytorchvideo.data import encoded_video
from torch.functional import norm
import torchaudio
from torch.nn import functional as F
from torchvision import transforms
from utils import normalize, unnormalize
from PIL import Image
from model import AdaINModel
import cv2
from tqdm import tqdm
import glob
import subprocess

# def get_images_result(x_norm):
#     im_np = np.uint8(255 * torch.clip(unnormalize(x_norm), 0., 1.).permute(0, 2, 3, 1).cpu().numpy())
#     return [x for x in im_np]

def get_images_result(x_norm, percentiles=[0, 98]):
    pixvals = x_norm.permute(0, 2, 3, 1).cpu().numpy()

    minval = np.percentile(pixvals, percentiles[0], axis=(1, 2, 3), keepdims=True)
    maxval = np.percentile(pixvals, percentiles[1], axis=(1, 2, 3), keepdims=True)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = np.uint8(((pixvals - minval) / (maxval - minval)) * 255)
    
    return [x for x in pixvals]

IM_SIZE = 512
transform = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    normalize,
])

encoded_vid = encoded_video.EncodedVideo.from_path("test_images/test.mp4")
im_style_bank = [
    transform(Image.open(im_p).convert('RGB')).unsqueeze(0) 
    for im_p in glob.glob("./test_images/style/*")
]

device = 'cuda:0'

model = AdaINModel()
weights_dict = torch.load(
    "/media/totolia/datos_3/research/adain/results/model_without_remove/25_09_2021__17_50_58/checkpoints/checkpoint_29_1.237152.pkl"
)
model.load_state_dict(weights_dict)
model.eval()
model.to(device)

video_save_path = 'output.mp4'
video_output = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 7, (640, 304))

for i in tqdm(range(int(np.floor(encoded_vid.duration)))):
    video_block = encoded_vid.get_clip(i, i + 1)
    video = video_block['video'].permute(1, 0, 2, 3)
    audio = video_block['audio']

    # Aggrupate audio
    # audio = audio[:(audio.shape[0] // 25 * 25)]
    # audio_chunks, _ = torch.max(audio.reshape(video.shape[0], -1), dim=-1)

    audio_chunks = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio.shape[0], 
        n_fft=4*400,
        hop_length=int(np.ceil(audio.shape[0] / video.shape[0])),
        n_mels=5
    )(audio)
    r = audio_chunks
    audio_chunks = (torchaudio.transforms.AmplitudeToDB()(audio_chunks[0, :]) + 100.0) / 170.
    # audio_chunks = torch.clip(audio_chunks[0, :], 0., 70.) / 70. # torchaudio.transforms.AmplitudeToDB()(audio_chunks)[:, 0].mean(dim=-1)

    # Resize video
    with torch.no_grad():
        video_resize = F.interpolate(video, size=[IM_SIZE, IM_SIZE])
        video_resize = normalize(video_resize.float() / 255.)

        styles_selected = torch.cat([im_style_bank[int(len(im_style_bank) * value)] for value in audio_chunks], axis=0)

        im_styled = model(
            video_resize.to(device), 
            styles_selected.to(device), 
            alpha=0.75
        )
        im_styled = F.interpolate(im_styled, size=[video.shape[2], video.shape[3]])

        im_styled_np = get_images_result(im_styled)
        for j in range(video.shape[0]):
            video_output.write(im_styled_np[j])

            #cv2.imwrite(f'output/{i * video.shape[0] + j}.jpg', im_styled_np[j])

video_output.release()
p = subprocess.Popen("ffmpeg -i output.mp4 -i test_images/test.mp4 -c copy -map 0:0 -map 1:1 -shortest output_audio.mp4", shell=True)
p.wait()

