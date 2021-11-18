import timm
from timm.data import ImageDataset, create_loader, resolve_data_config
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from collections import defaultdict
import os

if __name__ == '__main__':
    m = timm.create_model(model_name="vit_base_resnet50d_224", pretrained=True, checkpoint_path="checkpoints/resnet50d_vit.pth.tar")
    #m = timm.create_model(model_name="resnest101e", pretrained=True,checkpoint_path="checkpoints/resnest101e_best.pth.tar")
    m.eval()
    rootdir = "fly_images"
    # go through each image in the test set
    # find prob and verify it matches
    # get best 5 images for each species.
    with open("classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    #top_5 = defaultdict(list)

    prob_bracket = defaultdict(int)
    categories_dict = defaultdict(int)
    # for bee in categories:
    #     #top_5[bee] = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    #     top_5[bee] = [(0, 0)] * 10
    # print(top_5)
    config = resolve_data_config({}, model=m)
    transform = create_transform(**config)
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:

            img_dir = subdir + "/" + file + ""

            im = Image.open(img_dir).convert('RGB')
            im2 = transform(im).unsqueeze(0)

            with torch.no_grad():
                out = m(im2)
            probabilities = torch.nn.functional.softmax(out[0], dim=0)

            top5_prob, top5_catid = torch.topk(probabilities, 1)

            category = categories[top5_catid[0]]
            prob = top5_prob[0].item()
            rounded_prob = round(prob*10)*10
            prob_bracket[rounded_prob] += 1
            if prob > 0.90:
                categories_dict[category] += 1
                count += 1
    print(count)
    print(categories_dict)
    print(prob_bracket)
            # prob_check = [prob]
            # for i in top_5[category]:
            #     prob_check.append(i[0])
            #
            # pos = sorted(prob_check).index(prob)
            # if pos > 0:
            #     pos = pos - 1
            #
            #     top_5[category].append((prob, file))
            #     top_5[category].remove(top_5[category][0])
            #     top_5[category] = sorted(top_5[category], key=lambda x:x[0])

