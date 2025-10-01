import streamlit as st
from PIL import Image
import numpy as np
import torch 
import torch.nn as nn
import timm
from torchvision import transforms
import cv2

# page config 
st.set_page_config(page_title="ManuSpec Medical AI: Pneumonia Detection", layout="wide")

# load saved weights / model
# @st.cache_resource is a decorator that tells Streamlit to run this function only ONCE = 
# loading the model into memory and caching it. This prevents the model from being reloaded
# every time the user interacts with the app, which would be very slow.

@st.cache_resource
def load_model():
    #setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rebuilt the same model architecture 
    model = timm.create_model('efficientnet_b0', pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, 1)

    # load our saved weights into the model struct
    model.load_state_dict(torch.load('pneumonia_model.pth', map_location=device))

    # move the model to the selected device
    model.to(device)

    # set model to evaluation mode 
    model.eval()
    return model, device

model, device = load_model()

# image transformations 
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# grad cam logic 
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output 

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def generate_grad_cam(model, input_tensor, original_image):
    target_layer = model.conv_head
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output.backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * 0.4 + original_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    forward_handle.remove()
    backward_handle.remove()

    return superimposed_img, output
    
# header
st.title("ManuSpec Medical AI Pneumonia Detection")
st.write("Upload a chest X-Ray image and the AI model will analyze it for signs of pneumonia")

# sidebar and file uploader 
st.sidebar.header("Upload X-Ray")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # display uploaded img
    st.sidebar.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)

    #convert PIL Image into an opencv 
    pil_image = Image.open(uploaded_file).convert("RGB")
    opencv_image = np.array(pil_image)

    # main content areas
    st.write("---")
    st.header("Analysis")
    col1, col2 = st.columns(2)

    # convert the uploaded file to image that model can understand
    image_tensor = val_transform(pil_image).unsqueeze(0).to(device) 
    
    # get the model's prediction 
    superimposed_image, output = generate_grad_cam(model, image_tensor, opencv_image)
    
    # convert the output to a probability and then binary prediction
    prob = torch.sigmoid(output).item()
    prediction = 1 if prob > 0.5 else 0 

    with col1:
        st.subheader("Diagnosis: ")
        if prediction == 1:
            st.error(f"Pneumonia Detected (Confidence: {prob:.2%})", icon="⚠️")
        else:
            st.success(f"Normal (Confidence: {1-prob:.2%})", icon="✅")
        st.write("*(Model prediction will appear here)*")

    with col2: 
        st.subheader("Model's Focus (Heatmap):")
        superimposed_image_rgb = cv2.cvtColor(superimposed_image, cv2.COLOR_BGR2RGB)
        st.image(superimposed_image_rgb, caption="Heatmap shows areas of interest to the model.", use_container_width=True)

# Launch App: __NV__PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia streamlit run app.py

        
        