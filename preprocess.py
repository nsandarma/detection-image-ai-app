import torch
from torch import device, nn

from torchvision import transforms
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomCNN(nn.Module):

    def conv_block(in_channels,out_channels,kernel_size=3,padding=1,stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,padding,stride),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )

    def linear_block(in_features,out_features,dropout=0,activation=None):
        if activation == 'softmax':
            return nn.Sequential(nn.Linear(in_features,out_features),nn.LogSoftmax(1))
        return nn.Sequential(nn.Linear(in_features,out_features),nn.LeakyReLU(),nn.Dropout(dropout))

    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv = nn.Sequential(
            CustomCNN.conv_block(3,8),
            CustomCNN.conv_block(8,16),
            CustomCNN.conv_block(16,32),
            CustomCNN.conv_block(32,64),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            CustomCNN.linear_block(1024,128,dropout=0.5),
            CustomCNN.linear_block(128,64,dropout=0.5),
            CustomCNN.linear_block(64,2,activation='softmax')
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), 
        ])
    image = transform(image)
    return image.unsqueeze(0)

def load_image(im):
    im = Image.open(im)
    im = preprocess_image(im)
    return im

def load_model():
    model = CustomCNN()
    state_dict = torch.load('./model-airart-realart.pth',map_location=device)
    model.load_state_dict(state_dict)
    return model

def predict(im):
    feature = load_image(im)
    model = load_model()
    with torch.no_grad():
        model.eval()
        output = model.forward(feature)
        output = output.argmax(1).item()
        if output == 1:
            return "AI Image"
        else:
            return "Real Image"

if __name__ == '__main__':
    model = load_model()
    print(model.training)
