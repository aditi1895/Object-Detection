import torch
import os

BASE_PATH = "C:/Users/aditi/Downloads/Microsoft COCO.v2-raw.tensorflow"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "train"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

BASE_OUTPUT = "output"

MODEL_PATH = os.path.sep.join([BASE_PATH, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_PATH, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_PATH, "plots"])
TEST_PATHS = os.path.sep.join([BASE_PATH, "test_paths.txt"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE=="cuda" else False

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

LABELS = 1.0
BBOX = 1.0