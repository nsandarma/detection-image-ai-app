import torch
from torch import device, nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = {0:"Ai Image",1:"Real Image"}

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image.unsqueeze(0).to(device)

def load_image(im):
    im = Image.open(im)
    if im.mode == "RGBA":
        im  = im.convert("RGB")
    im = preprocess_image(im)
    return im

def load_model():
    model = resnet18()
    num_if = model.fc.in_features
    model.fc = nn.Linear(num_if,2)
    model.load_state_dict(torch.load("resnet_model.pth",map_location=device))
    return model.to(device)

def predict(im):
    feature = load_image(im)
    model = load_model()
    with torch.no_grad():
        model.eval()
        output = model.forward(feature)
        _,pred = torch.max(output,1)
        return class_names[pred.item()]
