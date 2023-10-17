import bm3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
def noisy(noise_typ,image):
   if image.shape.__len__() == 2:
      raise Exception("input image must be of 3 channels")
   if noise_typ == "gauss":
      row,col,ch = image.shape
      mean = 200
      var = 500
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[0:2]]
      out[coords[0], coords[1], 0] = 255

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[0:2]]
      out[coords[0], coords[1], 0] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy


# Read input image
path = "me.jpeg"
im = cv2.imread(path)
# Convert rgb image to grayscale
grayscale_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# Fix image dimensions
if grayscale_im.shape.__len__() == 2:
   grayscale_im = np.expand_dims(grayscale_im, axis=-1)
# Add noise to image
noisy_image = noisy("gauss", grayscale_im)
# Clean image using bm3d
clean_image = bm3d.bm3d(noisy_image, sigma_psd=50, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
# Plot original image in grayscale, noisy and bm3d cleaned
plt.figure()
im_plot = plt.imshow(grayscale_im, cmap="gray")
plt.figure()
noisy_image_plot = plt.imshow(noisy_image, cmap="gray")
plt.figure()
clean_image_plot = plt.imshow(clean_image, cmap="gray")
plt.show()

print("done")