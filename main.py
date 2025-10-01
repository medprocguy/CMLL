import numpy as np
import torch, random
import torch.nn as nn
from torchvision import transforms, models, ops
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataloader_coco import build
from tqdm import tqdm
from binning_fn_torch import canonical_ECE
import timm

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

class Corr_Loss(nn.Module):
    def __init__(self):
        super(Corr_Loss,self).__init__()
        self.eps = 1e-8
        
    def pearson_corr(self, mat):
        mat_centered = mat - mat.mean(dim = 0, keepdim= True)
        cov_matrix_mat = (mat_centered.T @ mat_centered)/(mat.shape[0] - 1 + self.eps) 
        std_mat = torch.sqrt(torch.diag(cov_matrix_mat) + self.eps) 
        corr_mat = cov_matrix_mat/(std_mat.unsqueeze(0).T @ std_mat.unsqueeze(0))
        
        return corr_mat 
        
    def forward(self, y_pred, labels):
        y_pred = y_pred.tanh()
        labels = 2 * labels - 1
        pred_corr = self.pearson_corr(y_pred)
        lbl_corr = self.pearson_corr(labels) 
        corr_diff = torch.triu(torch.abs(lbl_corr - pred_corr), diagonal=1)
        r,c = corr_diff.shape
        
        return corr_diff.sum()/torch.triu_indices(r, c, 1).shape[-1]
        
# define configs and create config
config = {
    # optimization configs
    'seed': 42,
    'epoch': 100,  
    'lambda': 1,
    'max_norm': 0.1,
    'batch_size': 32,
    'eval_batch_size': 32,
    'test_batch_size': 1,
    'num_classes': 14,
}

# fix the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

train_transform = transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(degrees=(-10, 10)),
                      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                      transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                      transforms.ToTensor()
                  ])
eval_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()
                 ])                  
                 
train_dataset, val_dataset = build(config, train_transform, eval_transform)

# Create DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['eval_batch_size'], shuffle=False)

# Load ResNet-50 model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, config['num_classes'])

# Load ViT b 32 model
# model = models.vit_b_32() 
# model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=config['num_classes'], bias=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion =  torch.nn.BCEWithLogitsLoss()
criterion_aux = Corr_Loss() 
best_val_loss = 9999
for epoch in range(config['epoch']):
    print('Epoch:',epoch,'/',config['epoch'])
    model.train()
    tr_loss, tr_bce, tr_aux, val_loss, val_bce, val_aux = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        bs = inputs.shape[0]
        
        optimizer.zero_grad()
        outputs = model(inputs.float())
                
        bce_loss = criterion(outputs, labels)
        aux_loss = criterion_aux(outputs, labels)   
        
        loss = bce_loss + config['lambda'] * aux_loss 
        
        assert not torch.isnan(loss)
        tr_bce += bce_loss.item()
        tr_aux += aux_loss.item()
        tr_loss += loss.item()
        loss.backward()
        if config['max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_norm'])
        optimizer.step()
       
    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            bs = inputs.shape[0]
            
            outputs = model(inputs.float())
            
            bce_loss = criterion(outputs, labels)
            aux_loss = criterion_aux(outputs, labels) 
        
            loss = bce_loss + config['lambda'] * aux_loss 
            val_bce += bce_loss.item()
            val_aux += aux_loss.item()  
            val_loss += loss.item() 
    
    val_loss = val_loss / len(val_loader)  
    print(f"Epoch {epoch+1}/{config['epoch']}, Loss_train: {tr_loss/len(train_loader)}, Loss_val: {val_loss}, BCE_loss_train: {tr_bce/len(train_loader)}, BCE_loss_val: {val_bce/len(val_loader)}, Corr_loss_train: {tr_aux/len(train_loader)}, Corr_loss_val: {val_aux/len(val_loader)}")
    
    save_on_master({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': config,
        'val_acc': val_loss,
    }, 'saved_data/latest_model.pth')       
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss    
        save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': config,
            'val_acc': val_loss,
        }, 'saved_data/best_model.pth')        
