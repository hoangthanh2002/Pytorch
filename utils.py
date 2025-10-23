import os as osp
import glob

def make_datapath_list(phase='train'):
    rootpath = './data/hymenoptera_data/'
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

path_list = make_datapath_list ('train')

train_img_list = make_datapath_list(phase='train')
val_img_list = make_datapath_list(phase='val')


def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()# eval là chế độ đánh giá, không cập nhật trọng số
            
            epoch_loss = 0.0 
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == 'train'):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):  
                    outputs = net (inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
