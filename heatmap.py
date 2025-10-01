import torch 
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import val_transform 
import os

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'pneumonia_model.pth'

# image path logic
test_pneumonia_dir = 'chest_xrays/chest_xray/test/PNEUMONIA'
image_files = os.listdir(test_pneumonia_dir)
first_image_name = image_files[0]
image_path = os.path.join(test_pneumonia_dir, first_image_name)
print(f"Analyzing image: {image_path}")

# load model and image
model = timm.create_model('efficientnet_b0', pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 1)

# load trained weights 
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# load and preprocess the img
original_img = cv2.imread(image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# the transform must be the same one used for validation/testing
transformed_img = val_transform(original_img)
# add a batch dimension 
input_tensor = transformed_img.unsqueeze(0).to(device)


# logic for generating the heatmap
target_layer = model.conv_head
activations = None
gradients = None 

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients 
    gradients = grad_out[0]

# register the hooks 
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_backward_hook(backward_hook)

# get model predictions
output = model(input_tensor)

# want to know which features were important for this specific output 
output.backward()

# compute the heatmap
# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
# weight the activation maps by importance 
for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]
# average the weighted activation maps along the channel dimension 
heatmap = torch.mean(activations, dim=1).squeeze()
# use relu to only keep positive contributions 
heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
# normalize the heatmap to be between 0 and 1 
heatmap /= np.max(heatmap)

# visualize the results, resize the heatmap to match the original image size 
heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
heatmap_colored = cv2.applyColorMap(np.uint8(225 * heatmap_resized), cv2.COLORMAP_JET)

# superimpose the heatmap onto the original image 
superimposed_img = heatmap_colored * 0.4 + original_img
superimposed_img = np.clip(superimposed_img, 0, 225).astype(np.uint8)

# display the images 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(original_img)
axes[0].set_title('Original X-Ray')
axes[0].axis('off')

axes[1].imshow(heatmap_resized, cmap='jet')
axes[1].set_title('Grad-CAM Heatmap')
axes[1].axis('off')

axes[2].imshow(superimposed_img)
axes[2].set_title('Superimposed Image')
axes[2].axis('off')

plt.show()












