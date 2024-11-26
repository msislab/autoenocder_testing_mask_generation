import os 
import numpy as np 
from model import *
from utils import  *
import cv2
from tqdm import tqdm 
from loguru import logger


autoencoder             = load_autoencoder()
dataset_test_auto      = 'Anomaly source dats set/bent'
os.makedirs("autoencoder_results",exist_ok=True)
try:
    for img_indx, img_file in enumerate(tqdm(os.listdir(dataset_test_auto),desc="running autoencoder testing")): 
        logger.info(f"Running Image {img_indx}")
        img_path                = os.path.join(dataset_test_auto,img_file)
        orig_img                = cv2.resize(cv2.imread(img_path),(256,256))
        input_image_np, reconstructed_image_np      = recons_image(img_path, autoencoder)
        final_results           = np.hstack((orig_img,reconstructed_image_np*255))
        cv2.imwrite(f"autoencoder_results/{img_indx}.png",final_results) # {img_file}
        
except Exception as e: print("Error is: ",e)