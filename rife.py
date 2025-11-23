import sys
sys.path.append('/home/belikan/Gabonextube/arXiv2021-RIFE')
sys.path.append('/home/belikan/Gabonextube/arXiv2021-RIFE/train_log')

from RIFE_HDv3 import Model
import torch
import numpy as np
from PIL import Image

class RIFE:
    def __init__(self):
        self.model = Model()
        self.model.load_model('/home/belikan/Gabonextube/arXiv2021-RIFE/train_log', -1)
        self.model.eval()
        self.model.device()

    def interpolate(self, frame1, frame2, num_frames=2):
        # frame1 and frame2 are PIL Images
        img0 = torch.from_numpy(np.array(frame1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img1 = torch.from_numpy(np.array(frame2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        inter = self.model.inference(img0.cuda(), img1.cuda())
        inter_np = (inter.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        inter_pil = Image.fromarray(inter_np)
        return [inter_pil] * num_frames