from time import time
import numpy as np
import matplotlib.pyplot as pylab
from dask import delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature


@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


# Load dataset
images = lfw_subset()
print(images.shape)
# (200, 25, 25)

# Display faces in a grid
fig = pylab.figure(figsize=(5, 5))
fig.subplots_adjust(left=0, right=0.9, bottom=0, top=0.9, hspace=0.05,
                    wspace=0.05)
for i in range(25):
    pylab.subplot(5, 5, i + 1), pylab.imshow(images[1+i, :, :], cmap='bone'),
pylab.axis('off')
pylab.suptitle('Faces')
pylab.show()

