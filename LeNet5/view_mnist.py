import struct
import numpy as np
import matplotlib.pyplot as plt

RAW_DIR = "./data/MNIST/raw"  # change if your data folder is elsewhere


def read_images(path):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"{path}: magic={magic}, images={count}, size={rows}x{cols}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_labels(path):
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        print(f"{path}: magic={magic}, labels={count}")
        return np.frombuffer(f.read(), dtype=np.uint8)


images = read_images(f"{RAW_DIR}/train-images-idx3-ubyte")
labels = read_labels(f"{RAW_DIR}/train-labels-idx1-ubyte")
print(images.shape)
# Show the first 25 training images with their labels
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for ax, img, lab in zip(axes.flat, images[:25], labels[:25]):
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {lab}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# Print one image as raw pixel numbers (0 = black, 255 = white)
print(f"\nRaw pixel values of image 0 (label {labels[0]}):")
np.set_printoptions(linewidth=200)
print(images[0])
