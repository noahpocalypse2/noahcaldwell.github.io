# ECE471_kmeans_clustering.py
# ECE471 Dr.Qi
# written by Noah Caldwell
# 4/1/19
# Implements k-means clustering and winner-takes-all to compress images.

import argparse
import numpy as np
import math
import random
from skimage import io, img_as_float
from decimal import Decimal


def MSE(A, B):
    # Returns mean-squared error of the compressed image
    return ( np.square(A - B) ).mean()

def PSNR(MSE):
    # Returns peak signal-to-noise ratio. Most easily defined via MSE
    return 20 * math.log10(255) - 10 * math.log10(MSE)

def k_means_clustering(image_vectors, k, num_iterations):
    # Create corresponding label array (Initialize with Label: -1)
    labels = np.full((image_vectors.shape[0],), -1)

    # Get arbitrary inital clusters
    cluster_prototypes = np.random.rand(k, 3)

    # Iteration loop
    for i in range(num_iterations):
        #print('Iteration: ' + str(i + 1))
        points_by_label = [None for k_i in range(k)]

        # Label them via closest point
        for rgb_i, rgb in enumerate(image_vectors):
            # [rgb, rgb, rgb, rgb, ...]
            rgb_row = np.repeat(rgb, k).reshape(3, k).T

            # Find the closest label via L2 norm
            closest_label = np.argmin(np.linalg.norm(rgb_row - cluster_prototypes, axis=1))
            labels[rgb_i] = closest_label

            if (points_by_label[closest_label] is None):
                points_by_label[closest_label] = []

            points_by_label[closest_label].append(rgb)

        # Optimize cluster prototypes (center of mass of cluster)
        for k_i in range(k):
            if (points_by_label[k_i] is not None):
                new_cluster_prototype = np.asarray(points_by_label[k_i]).sum(axis=0) / len(points_by_label[k_i])
                cluster_prototypes[k_i] = new_cluster_prototype

    return (labels, cluster_prototypes)


def winner_takes_all(image_vectors, k, num_iterations, learning=0.01):
    # Create corresponding label array
    labels = np.full((image_vectors.shape[0],), -1)

    # Assign centroids from random pixels in the image
    centroids = []
    for i in range(k):
        centroids.append(image_vectors[random.randint(0, len(image_vectors)-1)])

    # Iteration loop
    for i in range(num_iterations):
        #print("iteration " + str(i+1))

        for rgb_i, rgb in enumerate(image_vectors):
            rgb_row = np.repeat(rgb, k).reshape(3, k).T
            
            # Find which centroid is closest
            closest_label = np.argmin(np.linalg.norm(rgb_row - centroids, axis=1))
            labels[rgb_i] = closest_label

            # Nudge centroid towards the added point
            centroids[closest_label] = np.add(centroids[closest_label], learning * np.subtract(rgb, centroids[closest_label]))

    return (labels, centroids)

def compress_image(image_vectors, color_centroids, labels):
    output_image = np.zeros(image_vectors.shape)
    for i in range(output_image.shape[0]):
        output_image[i] = color_centroids[labels[i]]
    return output_image.reshape(image_dimensions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='image_compression', description='Image Compression', add_help=False)
    parser.add_argument('image_name', type=str, help='Image Filename')
    parser.add_argument('-k', type=int, dest='k', help='Number of Clusters', default=256)
    parser.add_argument('-e', type=float, dest='e', help='Learning Parameter', default=0.01)
    parser.add_argument('-i', '--iterations', type=int, dest='iterations', help='Number of Iterations', default=10)

    args = parser.parse_args()
    params = vars(args)

    image = io.imread(params['image_name'])[:, :, :3]  # Always read it as RGB (ignoring the alpha)
    image = img_as_float(image)

    image_dimensions = image.shape
    # Get image name without the extension
    image_tokens = params['image_name'].split('.')
    image_name = '.'.join(params['image_name'].split('.')[:-1]) if len(image_tokens) > 1 else params['image_name']

    # -1 infers dimensions from the length of the matrix, while keeping the last dimension a 3-tuple
    image_vectors = image.reshape(-1, image.shape[-1])

    k_means_labels, k_means_color_centroids = k_means_clustering(image_vectors, k=params['k'], num_iterations=params['iterations'])
    winner_takes_all_labels, winner_takes_all_color_centroids = winner_takes_all(image_vectors, k=params['k'], num_iterations=params['iterations'])

    #output_image = np.zeros(image_vectors.shape)
    #for i in range(output_image.shape[0]):
    #    output_image[i] = k_means_color_centroids[k_labels[i]]

    k_means_output_image = compress_image(image_vectors, k_means_color_centroids, k_means_labels)
    winner_takes_all_output_image = compress_image(image_vectors, winner_takes_all_color_centroids, winner_takes_all_labels)

    # Metrics
    mse = MSE(image,k_means_output_image)
    psnr = PSNR(mse)
    print("k-means (image above):\nMSE = {}\nPSNR = {}".format('%.2E' % Decimal(mse),'%.2E' % Decimal(psnr)))
    mse = MSE(image,winner_takes_all_output_image)
    psnr = PSNR(mse)
    print("winner-takes-all (image below):\nMSE = {}\nPSNR = {}".format('%.2E' % Decimal(mse),'%.2E' % Decimal(psnr)))
    
    print('k-means: saving compressed image...')
    io.imsave(image_name + '_compressed_with_k_means_' + str(params['k']) + '.png', k_means_output_image, dtype=float)
    print('winner-takes-all: saving compressed image...')
    io.imsave(image_name + '_compressed_with_winner_takes_all_' + str(params['k']) + '.png', winner_takes_all_output_image, dtype=float)
    print('Image Compression Completed!')
