from skimage import img_as_float
from skimage.io import imread
from skimage.transform import swirl
import matplotlib.pyplot as plt

im = img_as_float(imread(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\clock.jpg"))
swirled = swirl(im, rotation=0, strength=15, radius=200)
plt.imshow(swirled)
plt.axis('off')
plt.show()