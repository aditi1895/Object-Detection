from torch.nn import Dropout, Identity, Linear, Module, ReLU, Sequential, Sigmoid

class ObjectDetector(Module):

    def __init__(self, baseModel, numClasses, *args, **kwargs) -> None:
        super(ObjectDetector, self).__init__(*args, **kwargs)

        self.baseModel = baseModel
        self.numClasses = numClasses

        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128,64),
            ReLU(),
            Linear(64,32),
            ReLU(),
            Linear(32,4),
            Sigmoid()
        )

        self.classifier = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512,512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses)
        )

        self.baseModel.fc = Identity()

    def forward(self,x):
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)

        return(bboxes, classLogits)