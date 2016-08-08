import glob 
from skimage import io 
import numpy as np 

def load_image(file_path):
  imdata = io.imread(file_path)
  return imdata

def load_files(directory,batch):
  data= [load_image(image).astype(np.float32) for image in glob.glob(directory +"/*.png")[:batch]]
  return data	
