import requests 
import urllib
import numpy as np

def get_tiles(x_range,y_range,z,base):
  for x in np.arange(x_range[0], x_range[1]):
    for y in np.arange(y_range[0], y_range[1]):
      print "grabbing :", base.format(x=x,y=y,z=z)
      urllib.urlretrieve (base.format(x=x,y=y,z=z), "tiles/{x}_{y}_{z}.png".format(x=x,y=y,z=z))
x_range = (157,301)
y_range = (354,491)

base = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{x}/{y}'
get_tiles(x_range,y_range,10, base)
