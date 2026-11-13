import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models.mobilenetv2 import mobilenet_v2

from datetime import datetime           #File timstamp
import time                             #T+121239
from PIL import Image                   #Needed to load random image
import cv2 as cv                        #Show images and stuff(Can also be done with PIL
import numpy as np                      #Honestly this should just be a core library

TrainingRate = 0.01
EarlyStop = 10
AnnotationLevel = "family"
AnnotationFile = "./datasets/fgvc-aircraft-2013b/data/families.txt"
Timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

PreProcess = v2.Compose([                   #Preprocessing steps to get dataset to appropriate format
    #v2.Resize(224),                         #Resizes to 224px short side
    v2.Resize(None, max_size=224),          #Resizes to 224px long side
    v2.Pad(40),
    v2.CenterCrop(224),                     #Crops excess on long side
    v2.ToImage(),                           #Makes it an image???
    v2.ToDtype(torch.float32, scale = True) #Changes datatype to float and value range from 0-255 to 0-1
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    #Normalizes image because apparently the (pretrained)model likes that
])
model = mobilenet_v2(weights=None)          #This model wants data in the format of Color, Y, X and i hate it

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

annotations = []
with open(AnnotationFile, 'rb') as file:
    for line in file:
        annotations.append(line.decode("utf-8").strip())
        #print(annotations[-1])

TrainData = datasets.FGVCAircraft(root="./datasets", split="train", annotation_level=AnnotationLevel, download=True, transform=PreProcess)
TestData = datasets.FGVCAircraft(root="./datasets", split="test", annotation_level=AnnotationLevel, download=True, transform=PreProcess)
TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=80, shuffle=True, pin_memory=True)#, num_workers=args.nThreads
TestDataLoader = torch.utils.data.DataLoader(TestData, batch_size=80, shuffle=True, pin_memory=True)#, num_workers=args.nThreads

# for i in range(5):
    # im, label = TrainData[i]
    # im = (im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    # cv.putText(im, f"Label: {annotations[label]}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    # cv.imshow(f"{i}", im)
# cv.waitKey(0)

#criterion = nn.BCELoss()                #Binary classification
criterion = nn.CrossEntropyLoss()       #Multiclass classification
optimizer = torch.optim.Adam(model.parameters(), lr=TrainingRate)
def TrainEpoch():                       #Trains once on all training data
    accumulatedLoss = 0.0
    #Train on the whole dataset but tragically my GPU doesnt fit the whole dataset
    #(which is weird because its only a few GB) so it has to be in batches
    for BatchID, Batch in enumerate(TrainDataLoader):
        BatchIm, BatchLabel = Batch
        BatchIm, BatchLabel = BatchIm.to(device), BatchLabel.to(device)     #Put stuff on GPU(if cuda is available)
        
        optimizer.zero_grad()                   #Reset gradients
        output = model(BatchIm)                 #Make predictions
        loss = criterion(output, BatchLabel)    #How wrong are we
        loss.backward()                         #Change model weights based on how wrong we are
        optimizer.step()
        accumulatedLoss += loss.item()
    return accumulatedLoss / TrainDataLoader.batch_size

def TrainEpochs(Epochs):
    BestTest = np.inf
    NoImprovement = 0
    tStart = time.time()
    for Epoch in range(Epochs):
        model.train(True)
        print(f"Train------")
        LossTrain = TrainEpoch()
        
        LossTest = 0.0
        model.eval()
        with torch.no_grad():
            print(f"Eval-------")
            for BatchID, Batch in enumerate(TestDataLoader):
                BatchIm, BatchLabel = Batch
                BatchIm, BatchLabel = BatchIm.to(device), BatchLabel.to(device)     #Put stuff on GPU(if cuda is available)
                output = model(BatchIm)                                             #Make predictions
                LossTest += criterion(output, BatchLabel)                           #How wrong are we
                
            LossTest = LossTest / TestDataLoader.batch_size
        
        if LossTest < BestTest:
            BestTest, NoImprovement = LossTest, 0
            torch.save(model.state_dict(), f"Model_{Timestamp}_{Epoch+1}")
        else:
            NoImprovement += 1
            if NoImprovement > EarlyStop:
                print(f"Training does not continue until morale improves")
                break
        print(f"Epoch [{Epoch+1:>3}/{Epochs:>3}] T+{time.time() - tStart:>5.0f}, Training Loss: {LossTrain:>6.3f}, Test Loss: {LossTest:>6.3f},  Worse then best [{NoImprovement:>2}/{EarlyStop:>2}]")
TrainEpochs(1)

# confusion = np.zeros((len(annotations), len(annotations)))
# for BatchID, Batch in enumerate(TestDataLoader):
    # BatchIm, BatchLabel = Batch
    # BatchIm, BatchLabel = BatchIm.to(device), BatchLabel.to(device)
    # output = model(BatchIm)
    
    # probabilities = nn.functional.softmax(output, dim=1)
    # for i in range(len(probabilities)):
        # topProb, topClass = torch.topk(probabilities[i], 1)
        # confusion[BatchLabel[i].item()][topClass.item()] = topProb
# print(confusion)

testImage = Image.open("./a350.jpg")
testTensor = PreProcess(testImage).unsqueeze(0).to(device)  #Put the image through the preprocessing, then unsqueeze to add an extra dimension(image is 3D this model wants 4D

probabilities = nn.functional.softmax(model(testTensor), dim=1).squeeze()
topProb, topClass = torch.topk(probabilities, 5)
for i in range(len(topProb)):
    print(f"I am {topProb[i].item():.2f} percent sure this is a {annotations[topClass[i].item()]}")
# for i in range(5):
    # im, label = TrainData[i]
    # im = (im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # print(im.shape)
    # cv.putText(im, f"Label: {annotations[label]}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv.imshow(f"{i}", im)