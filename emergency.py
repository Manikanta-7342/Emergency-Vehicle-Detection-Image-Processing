import sys

from PIL import Image,ImageFilter

import time

from torchvision import *

from torch import *

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid


# east_path="D:\\Centuriton\\models\\Test\\test1\\Emergency\\6.0.720.mp4"


tick = time.time()
class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        # _,out = torch.max(out,dim = 1)
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets)
        return loss

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))


class Densenet169(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.pretrained_model = models.densenet169(pretrained=True)

        feature_in = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Linear(feature_in, 2)

    def forward(self, x):
        return self.pretrained_model(x)


imsize = (224, 224)
frames_to_skip = 10
li = []
im_output = "./"
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

# loading the model
loaded_densenet169 = Densenet169()
loaded_densenet169.load_state_dict(torch.load('./' + 'densenet169.pt', map_location=torch.device('cpu')))
loaded_densenet169.eval()


def predict_emergency_vehicle(image_path):
    uploaded_file = image_path

    image = Image.open(uploaded_file).convert('RGB')

    image = image.resize((224,224))

    image = image.filter(ImageFilter.MedianFilter)

    image = transform(image).view(1, 3, 224, 224)

    pred = loaded_densenet169.forward(image)
    proba, idx = torch.max(torch.sigmoid(pred), dim=1)

    proba = proba.detach().numpy()[0]
    idx = idx.numpy()[0]
    with open('emer.txt', 'w') as file:
        file.write("Emergency_Vehicle\nConfidence Level:"+str(float(proba))) if idx == 1 else file.write("Non_Emergency Vehicle\nConfidence Level:"+str(float(proba)))



predict_emergency_vehicle(sys.argv[1])


print('\n\n', 'Time taken: ', time.time() - tick)
