from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model , params_to_update, load_model
from dataset import Mydataset

def main ():
    train_dataset = Mydataset(file_list=train_img_list, transform =ImageTransform(size, mean, std), phase='train') 
    val_dataset = Mydataset(file_list=val_img_list, transform =ImageTransform(size, mean, std), phase='val')

    #dataloader
    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = { "train" : train_dataloader , "val" : val_dataloader}

    #network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    criterior = nn.CrossEntropyLoss()

    params1, params2, params3 = params_to_update (net)

    print(params_to_update)
    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4},
        {'params': params2, 'lr': 5e-4},
        {'params': params3, 'lr': 1e-3}]
    ,momentum=0.9) #lr là learning rate 

    train_model(net, dataloader_dict, criterior, optimizer, num_epochs=num_epochs)

if __name__ == '__main__':
    #main()
    #network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    load_model(net, save_path)