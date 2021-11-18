import PIL
from gradcam import GradCAM
from torchvision.models import resnet50
import timm
from timm.data import ImageDataset, create_loader, resolve_data_config
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import cv2
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image

#from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from utils import visualize_cam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_dir = 'testimages'
# img_name = 'collies.JPG'
# img_name = 'multiple_dogs.jpg'
# img_name = 'snake.JPEG'
img_name = 'bombus_griseocollis.jpg'
img_path = os.path.join(img_dir, img_name)
root = "./apis_xylocopa_bombus_11_dataset/val/xylocopa_virginica"

vgg = models.vgg11_bn(num_classes=11)
checkpoint = torch.load("./model_100_epochs_vgg.pt")
vgg.load_state_dict(checkpoint['model_state_dict'])
vgg.eval()
configs = [
    dict(model_type='vgg', arch=vgg, layer_name='features_28')
]
for config in configs:
    config['arch'].to(device).eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]
img = '46169602.jpg'
pil_img = PIL.Image.open(root + "/" + img).convert('RGB')
torch_img = transforms.Compose([
transforms.Resize((448, 448)),
transforms.ToTensor()])(pil_img).to(device)
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
torch_img_copy = torch_img.detach().clone()
for gradcam, gradcam_pp in cams:
    # mask, _ = gradcam(normed_torch_img)
    # heatmap, result = visualize_cam(mask, torch_img)
    torch_img_copyy = torch_img_copy.detach().clone()
    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img_copyy)
transforms.ToPILImage()(torch_img_copyy).save("researchimg" + "/" + "xylocopa_virginica " + img + "_cropped.jpg")