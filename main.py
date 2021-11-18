# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    m = timm.create_model(model_name="vit_base_resnet50d_224", pretrained=True, checkpoint_path="checkpoints/resnet50d_vit.pth.tar")
    m.eval()

    img_dir = "fly_images/19679338.jpg"
    # img_dir = "fly_images/apis_tester.jpg"
    img_out = "imagemaps/resnest101e_randaug_bombus_griseocollis_map.jpg"
    config = resolve_data_config({}, model=m)
    transform = create_transform(**config)
    im = Image.open(img_dir)
    im2 = transform(im).unsqueeze(0)

    with torch.no_grad():
        out = m(im2)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    with open("classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    print(categories)
    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print(top5_prob, top5_catid)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
        try:
            print(categories[top5_catid[i]], top5_prob[i].item())
        except:
            print("index")

    # heatmap = m.forward_features(im2).detach()
    #
    #
    #
    # width, height = im.size
    # #heatmap = o[0, :, :, 0]
    # #heatmap = heatmap.detach().numpy()
    # heatmap = torch.mean(heatmap, dim=1).squeeze()
    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= torch.max(heatmap)
    # plt.matshow(heatmap.squeeze())
    # plt.show()
    # heatmap = heatmap.numpy()
    # img = cv2.imread(img_dir)
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = heatmap * 0.6 + im
    # cv2.imwrite(img_out, superimposed_img)


    # image = plt.imread("62476303.jpg")
    # plt.imshow(image)
    # heatmap = heatmap.detach().numpy()
    # #plt.matshow(heatmap.squeeze())
    #
    # plt.imshow(heatmap, alpha=0.5, cmap="RdBu")
    # plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
