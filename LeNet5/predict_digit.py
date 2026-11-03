import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from lenet5_mnist import LeNet5  # same folder as this script

device = "cpu"
model = LeNet5().to(device)
model.load_state_dict(torch.load("lenet5_mnist.pt", map_location=device))
model.eval()


def preprocess(img):
    """Turn any image into the same format MNIST uses: white digit on black,
    centered in a 20x20 box inside 28x28, then padded to 32x32."""
    arr = np.array(img.convert("L"))
    if arr.mean() > 127:          # black ink on white paper -> flip it
        arr = 255 - arr
    ys, xs = np.where(arr > 50)
    if len(ys) == 0:
        return None
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20 / max(h, w)
    small = Image.fromarray(crop).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
    )
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - small.width) // 2, (28 - small.height) // 2))
    canvas = canvas.resize((32, 32), Image.BILINEAR)
    t = torch.from_numpy(np.array(canvas)).float() / 255
    t = (t - 0.1307) / 0.3081
    return t.unsqueeze(0).unsqueeze(0)


def predict(img):
    x = preprocess(img)
    if x is None:
        return None, 0.0
    with torch.no_grad():
        probs = F.softmax(model(x.to(device)), dim=1)[0]
    digit = int(probs.argmax())
    return digit, float(probs[digit]) * 100


def run_file(path):
    digit, conf = predict(Image.open(path))
    if digit is None:
        print("Couldn't find a digit in that image.")
    else:
        print(f"Prediction: {digit}  ({conf:.1f}% confident)")


def run_drawing_window():
    import tkinter as tk

    size = 280
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    root = tk.Tk()
    root.title("Draw a digit")
    canvas = tk.Canvas(root, width=size, height=size, bg="black")
    canvas.pack()
    result = tk.Label(root, text="Draw a digit, then click Predict", font=("Arial", 14))
    result.pack(pady=6)

    def paint(event):
        r = 10
        canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                           fill="white", outline="white")
        draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)

    def do_predict():
        digit, conf = predict(img)
        if digit is None:
            result.config(text="Nothing drawn yet")
        else:
            result.config(text=f"Prediction: {digit}  ({conf:.1f}% confident)")

    def clear():
        canvas.delete("all")
        draw.rectangle([0, 0, size, size], fill=0)
        result.config(text="Draw a digit, then click Predict")

    canvas.bind("<B1-Motion>", paint)
    row = tk.Frame(root)
    row.pack(pady=4)
    tk.Button(row, text="Predict", command=do_predict).pack(side="left", padx=4)
    tk.Button(row, text="Clear", command=clear).pack(side="left", padx=4)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        run_drawing_window()
