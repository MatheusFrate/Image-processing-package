from skimage.io import imread, imsave

def read_image(image_path, is_gray=False):
   image = imread(image_path, as_gray=is_gray)
   return image

def save_image(image, image_path):
   imsave(image_path, image)
    