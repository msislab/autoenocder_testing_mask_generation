from model import *
import torch
from PIL import Image
from torchvision import transforms

def load_autoencoder(weight_path='weights/autoencoder_model_weights_constant_regions.pth'):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    autoencoder = Autoencoder()  # Make sure this matches the model architecture
    autoencoder.load_state_dict(torch.load(weight_path, map_location=device))
    #autoencoder_weights/autoencoder_model_weights_front_tele.pth
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder

def recons_image(img_path, autoencoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to the input size
    transforms.ToTensor()           # Convert to a tensor
    ])
    try: image = Image.open(img_path).convert("RGB")
    except: image = Image.fromarray(img_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        reconstructed_image = autoencoder(input_image)
    
    input_image_np          = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed_image_np  = reconstructed_image.cpu().squeeze().permute(1, 2, 0).numpy()
    
    return input_image_np, reconstructed_image_np