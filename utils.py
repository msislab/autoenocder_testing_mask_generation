from model import *
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def load_autoencoder(weight_path='weights/autoencoder_model_weights_constant_regions.pth'):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    autoencoder = Autoencoderv1()  #Autoencoder() Make sure this matches the model architecture
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

def load_model_for_inference(model, path="vae_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

def reconstruct_image(model, input_image):
    model.eval()  # Ensure the model is in evaluation mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_image = input_image.to(device)
    
    with torch.no_grad():
        # Encode the input image to get the latent vector
        mu, logvar = model.encode(input_image)
        # Reparameterize (optional, for stochastic reconstruction)
        z = model.reparameterize(mu, logvar)
        # Decode the latent vector to reconstruct the image
        reconstructed_image = model.decode(z)
    
    return reconstructed_image

def plot_images(original, reconstructed,img_index=''):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(original.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"autoencoder_results/{img_index}.png")
    plt.show()

def load_image(image_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image
