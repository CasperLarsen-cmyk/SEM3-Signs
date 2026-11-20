from os import listdir, makedirs
from os.path import isdir
from PIL import Image, ImageOps

DatasetPath = "./OURVandalism - copy/"
#DatasetPath = "./datasets/GTSDB/images/val/"
imagePath = DatasetPath+""

imageList = []
filenames = listdir(imagePath)
for filename in filenames:
    imageList.append(DatasetPath+filename)

for file in filenames:
    image = Image.open(DatasetPath+file)
    image = ImageOps.exif_transpose(image)
    ow, oh = image.size
    scale = .25
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    output = Image.new(image.mode, (ow, oh), (114, 114, 114))
    output.paste(image, (int((ow - image.width) / 2), int((oh - image.height) / 2)))
    output.save(DatasetPath+file)