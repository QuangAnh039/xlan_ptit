import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as pylab


def plot_image(im, title):
    """
    Helper function to plot an image with a title.
    """
    pylab.imshow(np.array(im))
    pylab.title(title)


# Main code
i = 1
pylab.figure(figsize=(25, 35))

# Loop over noise proportions
for prop_noise in np.linspace(0.05, 0.3, 3):
    im = Image.open(r"C:\Users\Dell\Desktop\Sandipan_Dey_2018_Sample_Images\images\sea_bird.jpg")
    # Add salt-and-pepper noise to the image
    n = int(im.width * im.height * prop_noise)
    x, y = np.random.randint(0, im.width, n), np.random.randint(0, im.height, n)
    for (px, py) in zip(x, y):
        im.putpixel((px, py), ((0, 0, 0) if np.random.rand() < 0.5 else (255, 255, 255)))  # Add noise

    im.save(f'C:/Users/Dell/Desktop/Sandipan_Dey_2018_Sample_Images/images/mandrill_spnoise_{prop_noise:.2f}.jpg')

    # Plot noisy image
    pylab.subplot(6, 4, i)
    plot_image(im, f'Original Image with {int(100 * prop_noise)}% added noise')
    i += 1

    # Apply median filter with different sizes
    for sz in [3, 7, 11]:
        im1 = im.filter(ImageFilter.MedianFilter(size=sz))
        pylab.subplot(6, 4, i)
        plot_image(im1, f'Output (Median Filter size={sz})')
        i += 1
pylab.subplots_adjust(wspace=0.5, hspace=0.8)  # Tăng khoảng cách giữa các subplot
pylab.show()