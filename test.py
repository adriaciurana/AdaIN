import os
import glob
import argparse
import torch
from utils import normalize, unnormalize
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from model import AdaINModel

def get_images_result(x_norm):
    im_np = np.uint8(255 * torch.clip(unnormalize(x_norm), 0., 1.).permute(0, 2, 3, 1).cpu().numpy())
    return [Image.fromarray(x) for x in im_np]

def create_grid(images_content, images_style, output_dict, im_size):
    rows = len(images_content)
    cols = len(images_style)

    mosaic = np.ones((
        (1 + rows) * im_size,
        (1 + cols) * im_size,
        3
    ), dtype=np.uint8)
    
    for i, im in enumerate(images_content):
        mosaic[
            (i + 1) * im_size:(i + 2) * im_size, 
            0:im_size
        ] = np.asanyarray(im)

    for j, im in enumerate(images_style):
        mosaic[
            0:im_size, 
            (j + 1) * im_size:(j + 2) * im_size
        ] = np.asanyarray(im)

    for (i, j), im in output_dict.items():
        mosaic[
            (i + 1) * im_size:(i + 2) * im_size, 
            (j + 1) * im_size:(j + 2) * im_size
        ] = np.asanyarray(im)

    return Image.fromarray(mosaic)

def create_mosaic(images_content_paths, images_style_paths, num_batchs=5, device='cpu', im_size=512):
    images_content = [Image.open(im_c_p).convert('RGB').resize((IM_SIZE, IM_SIZE)) for im_c_p in images_content_paths]
    images_style = [Image.open(im_c_s).convert('RGB').resize((IM_SIZE, IM_SIZE)) for im_c_s in images_style_paths]

    def __internal_generator():
        for i, im_c_p in enumerate(images_content_paths):
            im_content_pil = Image.open(im_c_p).convert('RGB')
            im_content = transform(im_content_pil).unsqueeze(0)

            for j, im_s_p in enumerate(images_style_paths):
                im_style_pil = Image.open(im_s_p).convert('RGB')
                im_style = transform(im_style_pil).unsqueeze(0)

                yield (
                    i, im_content,
                    j, im_style,
                )

    output_dict = {}
    iter_gen = iter(tqdm(__internal_generator(), desc="Computing pairs..."))
    with torch.no_grad():
        is_end = False
        while not is_end:
            # Make an interation
            indices_pairs = []
            x_content = [] 
            x_style = [] 
            for i in range(num_batchs):
                try:
                    (
                        i, im_content,
                        j, im_style,
                    ) = next(iter_gen)
                    indices_pairs.append((i, j))

                    x_content.append(im_content)
                    x_style.append(im_style)

                except StopIteration:
                    is_end = True
                    continue

            # Model
            x_content = torch.cat(x_content, dim=0).to(device)
            x_style = torch.cat(x_style, dim=0).to(device)
            x_norm = model(x_content, x_style, alpha=args.alpha)
            images_styled = get_images_result(x_norm)
            for (i, j), im_styled in zip(indices_pairs, images_styled):
                output_dict[(i, j)] = im_styled

    # create mosaic
    return create_grid(
        images_content,
        images_style,
        output_dict,
        im_size=im_size
    )


                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference algorithm')
    parser.add_argument('-w', '--weights_path', type=str, help='Weights of the model', required=True)
    parser.add_argument('-ic', '--image_content_path', type=str, help='Image content', required=True)
    parser.add_argument('-is', '--image_style_path', type=str, help='Image style', required=True)
    parser.add_argument('-a', '--alpha', type=float, help='Alpha', required=False, default=.5)
    parser.add_argument('-d', '--device', type=str, help='Device for the training', default=0)
    args = parser.parse_args()

    DEVICE = torch.device("cuda:%d" % (args.device, ) if torch.cuda.is_available() else "cpu")
    IM_SIZE = 224

    transform = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # Model
    model = AdaINModel()
    weights_dict = torch.load(args.weights_path)
    model.load_state_dict(weights_dict, strict=True)
    model.eval()
    model = model.to(DEVICE)

    if os.path.isdir(args.image_content_path):
        images_content_paths = glob.glob(os.path.join(args.image_content_path, "*")) 

    else:       
        images_content_paths = [args.image_content_path]

    if os.path.isdir(args.image_style_path):
        images_style_paths = glob.glob(os.path.join(args.image_style_path, "*")) 

    else:       
        images_style_paths = [args.image_style_path]

    mosaic = create_mosaic(
        images_content_paths,
        images_style_paths,
        num_batchs=5,
        device=DEVICE,
        im_size=IM_SIZE
    )
    mosaic.save('mosaic.jpg')
                
                



