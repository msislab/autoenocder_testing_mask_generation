import os 
import numpy as np 
from model import *
from utils import  *
import cv2
from tqdm import tqdm 
from loguru import logger

model                   = VAE(latent_dim=128).to("cuda" if torch.cuda.is_available() else "cpu")
autoencoder             = load_model_for_inference(model, path="weights/vae_model.pth")
dataset_test_auto      = 'test_imgs'
os.makedirs("autoencoder_results",exist_ok=True)
try:
    for img_indx, img_file in enumerate(tqdm(os.listdir(dataset_test_auto),desc="running autoencoder testing")): 
        logger.info(f"Running Image {img_indx}")
        img_path                = os.path.join(dataset_test_auto,img_file)
        # Load an input image
        input_image = load_image(img_path)
        reconstructed_image = reconstruct_image(model, input_image)
        plot_images(input_image, reconstructed_image,img_index= img_indx)
            
except Exception as e: print("Error is: ",e)

    