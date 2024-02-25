

import torch
from torchvision import transforms
import os

finetune = False

data_root = 'drive/MyDrive/dataset1'
mode = 'train'
modelPath = 'drive/MyDrive/model1/modelFile1.pth'




# Define the used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
print("Loading the Model...")
model = PConvUNet(finetune=finetune,
                  layer_size=7)
if finetune:
    model.load_state_dict(torch.load(modelPath)['model'])
model.to(device)


# Data Transformation
img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size = (512,512))
            ])


mask_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size = (512,512))
        ])

# Define the Validation set
print("Loading the Validation Dataset...")
dataset_val = Places2(data_root,
                      img_tf,
                      mask_tf,
                      data="val")



# Set the configuration for training
if mode == "train":


    # Define the Places2 Dataset and Data Loader
    print("Loading the Training Dataset...")
    dataset_train = Places2(data_root,
                            img_tf,
                            mask_tf,
                            data="train")
                            
    initial_lr = 0.0002
    optim = "Adam"
    weight_decay = 0

    print("Length of dataset for train : ",len(dataset_train))




    # Define the Loss fucntion
    criterion = InpaintingLoss(VGG16FeatureExtractor(),
                               tv_loss='mean').to(device)
    # Define the Optimizer
    lr = initial_lr
    if optim == "Adam":
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                     lr=lr,
                                     weight_decay=weight_decay)
    elif optim == "SGD":
      optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           model.parameters()),
                                    lr=lr,
                                    momentum=0,
                                    weight_decay=weight_decay)

    start_iter = 0
    if os.path.exists(modelPath):
      print("Loading the trained params and the state of optimizer...")
      '''
      start_iter = load_ckpt([("model", model)],
                               [("optimizer", optimizer)])
                               '''

      checkpoint = (torch.load(modelPath))
      model.load_state_dict( checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      for param_group in optimizer.param_groups:
        param_group["lr"] = lr
      print("Starting from iter ", start_iter)

    trainer = Trainer(start_iter,  device, model, dataset_train,
                      dataset_val, criterion, optimizer)
    trainer.iterate()

