import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from model import PConvUNet

from torchvision import transforms

resize = transforms.Resize(size = (512,512))




# Define the used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the model
print("Loading the Model...")
model = PConvUNet(layer_size=7)
model.load_state_dict(torch.load('modelFile.pth', map_location=device)['model'])
model.to(device)
model.eval()

# Loading Input and Mask
print("Loading the inputs...")
print(os.getcwd())
os.chdir('D:\FINAL')



org = Image.open('ronaldo.jpg')

org = TF.to_tensor(org.convert('RGB'))
mask = Image.open('ronaldomask.png')
mask = TF.to_tensor(mask.convert('RGB'))



org = resize(org)
mask =  resize(mask)

inp = org * mask
# Model prediction
print("Model Prediction...")
with torch.no_grad():
    inp_ = inp.unsqueeze(0).to(device)
    mask_ = mask.unsqueeze(0).to(device)
    
    raw_out, raw_mask = model(inp_, mask_)

# Post process
raw_out = raw_out.to(torch.device('cpu'))
raw_out = raw_out.clamp(0.0, 1.0)
out = mask * inp + (1 - mask) * raw_out

plt.imshow(out.cpu().detach()[0].permute(1,2,0))
plt.show()

