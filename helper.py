import os
root = "./testdatasett"
for path, subdirs, files in os.walk(root):
    if not("_cropped") in path:
        species = path.split('\\')[-1]
        for image in files:
            print(image + species)
            print(path)


