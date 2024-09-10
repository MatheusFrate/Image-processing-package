import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity

def find_differeces(image1, image2):
    assert image1.shape == image2.shape, "As imagens tem que ter o mesmo tamanho"
    image1_gray = rgb2gray(image1)
    image2_gray = rgb2gray(image2)
    (score, difference_image) = structural_similarity(image1_gray, image2_gray, full=True)
    print("Similaridade das imagens: ", score)
    normalized_difference_image = (difference_image-np.min(difference_image))/(np.max(difference_image)-np.min(difference_image))
    return normalized_difference_image

def transfer_histogram(image1, image2):
    matched_image = match_histograms(image1, image2, multichannel=True)
    return matched_image
