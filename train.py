from bbox_regressor import ObjectDetector
from custom_tensor_dataset import CustomTensorDataset
from config import *
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch, time, cv2, os

def train():
    data = []
    labels = []
    bboxes = []
    imagePaths = []

    print("Loading Data")
    for csvPath in paths.list_files(ANNOTS_PATH, validExts=(".csv")):
        rows = open(csvPath).read().strip().split("\n")[1:1100]

        for row in rows: 
            row = row.split(",")
            # (filename, startX, startY, endX, endY, label) = row
            if len(row)<8 or row[0]=='':
                continue
            (filename,width, height,label,startX, startY, endX, endY) = row

            imagePath = os.path.sep.join([IMAGES_PATH, filename])

            image = cv2.imread(imagePath)
            (h,w) = image.shape[:2]

            startX = float(startX) / w
            startY = float(startY) / h
            endX = float(endX) / w
            endY = float(endY) / h

            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            data.append(image)
            labels.append(label)
            bboxes.append((startX, startY, endX, endY))
            imagePaths.append(imagePath)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    imagePaths = np.array(imagePaths)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.20, random_state=42)

    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    (trainPaths, testPaths) = split[6:]
    
    trainDS = CustomTensorDataset((trainImages, trainLabels, trainBBoxes), transforms=transforms)
    testDS = CustomTensorDataset((testImages, testLabels, testBBoxes), transforms=transforms)
    print("Train Data loader", trainDS, len(trainDS.tensors[0]))

    print(f"total training samples {len(trainDS)}")
    print(f"total training samples {len(testDS)}")

    trainSteps = len(trainDS) // BATCH_SIZE
    valSteps = len(testDS) // BATCH_SIZE

    trainLoader = DataLoader(trainDS)#, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)
    testLoader = DataLoader(testDS)#, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)

    f = open(TEST_PATHS, "w")
    f.write("\n".join(testPaths))
    f.close()

    resnet = resnet50(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    objectDetector = ObjectDetector(resnet, len(le.classes_))
    objectDetector = objectDetector.to(DEVICE)

    classLossFunc = CrossEntropyLoss()
    bboxLossfunc = MSELoss()

    opt = Adam(objectDetector.parameters(), lr=INIT_LR)
    # print(objectDetector)

    History = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [], "val_class_acc": []}

    startTime = time.time()

    for e in tqdm(range(NUM_EPOCHS)):
        objectDetector.train()

        totalTrainLoss = 0
        totalValLoss = 0

        trainCorrect = 0
        valCorrect = 0

        # for v in trainDS:
        #     print(v)
        for (images, labels, bboxes) in trainLoader:
            (images, labels, bboxes) = (images.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE))

            # print(images.shape)
            predictions = objectDetector(images)
            bboxLoss = bboxLossfunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)
            totalLoss = (BBOX*bboxLoss) + (LABELS*classLoss)

            opt.zero_grad()
            totalLoss.backward()
            opt.step()

            totalTrainLoss+=totalLoss
            trainCorrect+=(predictions[1].argmax(1)==labels).type(torch.float).sum().item()

        with torch.no_grad():

            objectDetector.eval()

            for (images, labels, bboxes) in testLoader:
                (images, labels, bboxes) = (images.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE))

                predictions = objectDetector(images)
                bboxLoss = bboxLossfunc(predictions[0], bboxes)
                classLoss = classLossFunc(predictions[1], labels)
                totalLoss = (BBOX*bboxLoss)+(LABELS*classLoss)

                totalValLoss+=totalLoss

                valCorrect+=(predictions[1].argmax(1)==labels).type(torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        trainCorrect = trainCorrect /len(trainDS)
        valCorrect = valCorrect / len(testDS)

        History["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        History["train_class_acc"].append(trainCorrect)
        History["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
        History["val_class_acc"].append(valCorrect)

        print(f"EPOCH: {e+1}/{NUM_EPOCHS}")
        print(f"Train loss {avgTrainLoss:.6f} Train Accuracy {trainCorrect: .4f}")
        print(f"Val loss {avgValLoss:.6f} Train Accuracy {valCorrect: .4f}")

    endTime = time.time()
    print(f"Total time taken {endTime - startTime}")


    print("Saving object Detector model...")
    torch.save(objectDetector, MODEL_PATH)

    print("saving label encoder")
    f = open(LE_PATH, "wb")
    f.write(pickle.dumps(le))
    f.close()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(History["total_train_loss"], label = "total_train_loss")
    plt.plot(History["total_val_loss"], label = "total_val_loss")
    plt.plot(History["train_class_acc"], label="train_class_acc")
    plt.plot(History["val_class_acc"], label="val_class_acc")
    plt.title("Total Training loss and Classification accuracy on Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc = "lower left")
    plt.show()


if __name__ == '__main__':
    train()