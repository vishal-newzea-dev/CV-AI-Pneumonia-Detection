import torch # pytorch
import torch.nn as nn # pytorch nn = neural networks
import torch.optim as optim #pytorch optimisers 
import timm # pytorch image models
from data_pipeline import train_loader, val_loader #importing custom data pipelines 

# define the number of times we will loop over the entire training dataset 
# each full pass is an epoch 
EPOCHS = 5

# automatically detect nvidia gpu - uses gpu if found otherwise uses cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load pre trained efficientnet-B0 model using timm. 
model = timm.create_model('efficientnet_b0', pretrained=True)

# replacing the classifier / final layer's output of 1000 classes for imagenet to a more
# suitable one for normal vs pneumonia tasks
num_features = model.classifier.in_features

# replace classifier with a linear layer - models prediction for the probablity of pneumonia 
model.classifier = nn.Linear(num_features, 1)

#move the entire model to the selected device 
model.to(device)

# loss function / optimizer 
# 'BCEWithLogitsLoss' is mathematically stable and designed for binary classification
# where the model outputs a single number (a logit). It combines a Sigmoid function
# and Binary Cross-Entropy loss in one step.

criterion = nn.BCEWithLogitsLoss()

# the optimizer is the algorithm that updates the model's weights to reduce the loss. 
# adam is an effective optimizer 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop where the learning happens
print("\nStarting training...")
print("About to enter main training loop...")
#specified number of epochs
for epoch in range(EPOCHS):
    #training phase
    print(f"--- Starting Epoch {epoch+1}/{EPOCHS} ---")
    model.train() #set model to training mode 
    running_loss = 0.0 

    # loop over each batch of data from our training data loader 
    for images, labels in train_loader:
        # move data for current batch to gpu/cpu
        images, labels = images.to(device), labels.to(device)
        # unsqueeze adds a dimension to labels tensor to match the models output shape 
        # changes the shape from [32] to [32, 1]
        labels = labels.unsqueeze(1)
        # zero the gradients 
        optimizer.zero_grad()
        # forward pass 
        outputs = model(images)
        # calculate the loss by comparing the models output to the true labels 
        loss = criterion(outputs, labels)
        # backward pass - calculate the gradients - the direction and magnitude of change required
        loss.backward()
        # update the weights: the optimizer takes a step in the direction of the gradients 
        optimizer.step()
        # add the loss from this batch to a running total
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # validation phase 
    model.eval() # set the model to evalulation mode 
    val_loss = 0.0 
    correct = 0
    total = 0

    # disable gradient calculation to speed up and save memory as not learning during validation 
    with torch.no_grad():
        # loop over each batch of data from our validation data loader 
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # convert the models logit outputs to probabilites using sigmoid 
            predicted_probs = torch.sigmoid(outputs)
            # get the final prediction 0 or 1 - by checking if probability is > 0.5
            predicted_labels = (predicted_probs > 0.5).float()
            # count the number of correct predictions in this batch 
            total += labels.size(0)
            correct+=(predicted_labels == labels).sum().item()
        
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    # print summary for epoch 
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Accuracy: {val_accuracy:.2f}%")
    
            
print("\nTraining complete!")
torch.save(model.state_dict(), 'pneumonia_model.pth')
print("Model saved to pneumonia_model.pth")
























            

























    

























        









        