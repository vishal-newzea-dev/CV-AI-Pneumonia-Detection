import torch 
import timm
from sklearn.metrics import classification_report, confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt 
import os
from data_pipeline import XRayDataset, val_transform # reuse the validation 

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# define path to test data
base_dir = 'chest_xrays/chest_xray'
test_dir = os.path.join(base_dir, 'test')

# model architecture 
model = timm.create_model('efficientnet_b0', pretrained=False) # load custom weights 
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 1)

# load trained weights 
model.load_state_dict(torch.load('pneumonia_model.pth'))
model.to(device) # model to gpu


# create the test data loader
test_dataset = XRayDataset(data_dir=test_dir, transform=val_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# the evaluation loop 
model.eval()
all_preds = []
all_labels = []

# no need to track gradients, using torch no grad for effiecency 
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # get the models raw output
        outputs = model(images)

        # convert logits to probabilites 
        predicted_probs = torch.sigmoid(outputs.squeeze())
        predicted_labels = (predicted_probs > 0.5).float()

        # store predictions and true labels to analyze later - moving to cpu to use with sklearn and numpy
        all_preds.extend(predicted_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# report the results 
print("\n" + "="*10)
print("CLASSIFICATION REPORT")
print("\n" + "="*10)
print(classification_report(all_labels, all_preds, target_names = ['Normal (0)', 'Pneumonia (1)']))
    
print("\n" + "="*10)
print("CONFUSION MATRIX")
print("\n" + "="*10)
cm = confusion_matrix(all_labels, all_preds)

# visualize confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal','Pneumonia'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.show()


      












        