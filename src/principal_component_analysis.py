import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path
from PIL import Image
import seaborn as sns
from skimage.transform import resize


def pca_svd(dataset_path):
    """
    This is a method to compute energies contained in principle components of the given dataset
    using SVD decomposition. We plot scatter plot of centered data and principle components.
    We also plot data along the principal components and then compare that with the plots of built
    in pca implementation to test our implementation.

    :param dataset_path: Path to the dataset in string format.
    :return: Returns a list of energies contained in the principle components in a decreasing order.
    """
    names = ['x', 'f(x)']
    dataset = pd.read_csv(dataset_path, sep=' ', names=names)
    centered_dataset = (dataset - dataset.mean())
    U, s, Vt = np.linalg.svd(centered_dataset, full_matrices=True)
    """
    # * U and V are the singular matrices, containing orthogonal vectors of
    #   unit length in their rows and columns respectively.
    # 
    # * S is a diagonal matrix containing the singular values of centered_dataset - these
    #   values squared divided by the number of observations will give the
    #   variance explained by each PC.
    """

    """
    # Following code computes the energies contained in each principal component.
    """
    variance = np.square(s)
    trace = np.sum(variance)
    energies = variance / trace
    print(energies)

    """
    # We are plotting a scatter plot of centered dataset along with the principal components.
    # Here ratio of lengths of arrows is merely for representation puposes and doesn't actually scale as per 
    # energies in each principal component. We chose to do this way because energy in PC2 <<< PC1 and the real
    # ratio value would make one of the PC arrows invisible.
    """
    ratio = 4 * s[1] / s[0]
    centered_dataset.plot(x='x', y='f(x)', kind='scatter', alpha=.5)
    plt.arrow(0, 0, Vt[0, 0], Vt[0, 1], alpha=1, length_includes_head=True, head_width=0.08, head_length=0.08)
    plt.arrow(0, 0, Vt[1, 0] * ratio, Vt[1, 1] * ratio, alpha=1, length_includes_head=True, head_width=0.08,
              head_length=0.08)
    plt.axis('equal')
    plt.title("Dataset Analyzed with PCA (Our Implementation)")
    plt.show()

    """
    # We are plotting the scatter plot of data along the principal components in the following code.
    """
    principalDf = centered_dataset @ Vt.T
    principalDf.columns = ['principal component 1', 'principal component 2']
    principalDf.plot(x='principal component 1', y='principal component 2', kind='scatter',
                     alpha=0.7, xlim=[-2, 2], ylim=[-.2, .2])
    plt.title("(Our Implementation) Dataset plotted against principal components")
    plt.show()


def pca_built_in(dataset_path):
    """
    This is a method to compute energies contained in principle components of the given dataset
    using built in PCA implementation. We are using this method to test the results of our SVD
    implementation. This method also plots data along principal components and we use this plot to
    compare with plot of our implementation. Matching plots signify that our implementation is correct.

    :param dataset_path: Path to the dataset in string format.
    :return: Returns a list of energies contained in the principle components in a decreasing order.
    """
    dataset = pd.read_csv(dataset_path, sep=' ')
    centered_dataset = (dataset - dataset.mean())
    pca = PCA()
    principalComponents = pca.fit_transform(centered_dataset)
    print(pca.explained_variance_ratio_)

    """
    # We are plotting the scatter plot of data along the principal components in the following code.
    """
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf.plot(x='principal component 1', y='principal component 2', kind='scatter',
                     alpha=0.7, xlim=[-2, 2], ylim=[-.2, .2])
    plt.title("(Built-in) Dataset plotted against principal components")
    plt.show()


def reconstruct_image(num_pc, U, s, Vt, explained_variance_ratio):
    """
    This method reconstructs the image from U, s, Vt matrix based on the given number of
    principal components to be used.

    :param num_pc: Number of principal components to be used for reconstruction.
    :param U:
    :param s:
    :param Vt:
    :param explained_variance_ratio: np array containing energy values for principal
            components in decreasing order.
    :return:
    """

    energy = np.sum(explained_variance_ratio[:num_pc])
    U = U[:, :num_pc]
    s = np.diag(s[:num_pc])
    Vt = Vt[:num_pc, :]
    reconstructed_img = U @ s @ Vt
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title(str(num_pc) + " Principal Components (" + str(energy * 100)+"%)")
    plt.show()


def pca_image(image_path):
    """
    This is a method for Task 1 part 2. This method plots several plots of reconstructed image of provided image path
    along with original image and a histogram representing energies along each pay component.

    :param image_path: Path to the image to be used.
    :return:
    """

    """
    # Load gray scaled image.
    """
    img_gray = Image.open(image_path).convert('LA')
    plt.imshow(img_gray)
    plt.title("Original Grayscaled Image")
    plt.show()

    image_matrix = np.array(list(img_gray.getdata(band=0)), float)
    image_matrix.shape = (img_gray.size[1], img_gray.size[0])

    """
    # resize image matrix to specified size as mentioned in exercise.
    """
    image_matrix = resize(image_matrix, (249, 185))
    plt.imshow(image_matrix, cmap='gray')
    plt.title("Resized Image")
    plt.show()

    """
    # center the values by subtracting mean.
    """
    image_matrix_centered = image_matrix - image_matrix.mean()
    U, s, Vt = np.linalg.svd(image_matrix_centered)
    """
    # calculate energies for each principal component
    """
    explained_variance_ratio = s ** 2 / np.sum(s ** 2)

    """
    # plot histogram representing the calculated energies for 20 highest principal components.
    """
    sns.barplot(x=list(range(1, 21)), y=explained_variance_ratio[0:20])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.tight_layout()
    plt.show()

    """
    # plot reconstructed images for principal component values in the list.
    # Apart from values mentioned in the exercise, we have plotted for
    # principal component values of 68, 69, 70 to show when the energy loss
    # becomes less than 1%.
    """
    for num_components in [10, 50, 68, 69, 120, len(s)]:
        reconstruct_image(num_components, U, s, Vt, explained_variance_ratio)


def pca_trajectory(dataset_path):
    """
    This is a method to plot and analyze of the provided dataset path.
    This method is specific to the requirements of Task 1 Third part.

    :param dataset_path: Path to dataset to be analyzed.
    :return:
    """

    ped_path = pd.read_csv(dataset_path, sep=' ').to_numpy()
    """
    # x and y coordinates of pedestrian 1 and 2. These correspond to first 4 columns of the dataset.
    # The following plot visualizes the path for first 2 pedestrians through a scatter plot.
    """
    ped_1_x = ped_path[:, 1]
    ped_1_y = ped_path[:, 2]
    ped_2_x = ped_path[:, 3]
    ped_2_y = ped_path[:, 4]

    plt.plot(ped_1_x.tolist(), ped_1_y.tolist(), 'r.', alpha=0.7)
    plt.plot(ped_2_x.tolist(), ped_2_y.tolist(), 'g.', alpha=0.7)
    plt.title("Pedestrian Paths")
    plt.show()

    """
    # centered x and y coordinates of pedestrian 1 and 2.
    # The following plot visualizes the path for first 2 pedestrians through a scatter plot
    # after the x and y values are centered depicting that the shape of the path doesn't change
    # for both pedestrians.
    """
    ped_1_x = ped_path[:, 1] - ped_path[:, 1].mean()
    ped_1_y = ped_path[:, 2] - ped_path[:, 2].mean()
    ped_2_x = ped_path[:, 3] - ped_path[:, 3].mean()
    ped_2_y = ped_path[:, 4] - ped_path[:, 4].mean()

    plt.plot(ped_1_x.tolist(), ped_1_y.tolist(), 'r.', alpha=0.7)
    plt.plot(ped_2_x.tolist(), ped_2_y.tolist(), 'g.', alpha=0.7)
    plt.title("Pedestrian Paths Centered")
    plt.show()


    # names_x = ['x'+str(i) for i in range(1, 16)]
    # names_y = ['y' + str(i) for i in range(1, 16)]
    # names = list(itertools.chain(*zip(names_x, names_y)))
    """
    # Following code centers the dataset and decomposes data using SVD.
    # We also calculate the energies along each of the 30 principal components
    # of data here.
    """
    ped_dataset = pd.read_csv(dataset_path, sep=' ')
    centered_dataset = (ped_dataset - ped_dataset.mean())
    U, s, Vt = np.linalg.svd(centered_dataset, full_matrices=True)
    explained_variance_ratio = s ** 2 / np.sum(s ** 2)

    """
    # We plot the energies along each principal component to visualize and analyze the data.
    """
    sns.barplot(x=list(range(1, len(explained_variance_ratio)+1)), y=explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.tight_layout()
    plt.title("Principal Component Energies")
    plt.show()

    """
    # We calculate energies along first 2 and first 3 principal components.
    """
    num_pc = 2
    energy2 = np.sum(explained_variance_ratio[:num_pc])
    print("Energy Stored in first 2 principal components: " + str(energy2))
    num_pc = 3
    energy3 = np.sum(explained_variance_ratio[:num_pc])
    print("Energy Stored in first 3 principal components: " + str(energy3))


if __name__ == '__main__':
    # Task 1 First Part
    cwd = Path.cwd()
    path = cwd/"datasets"/"pca_dataset.txt"
    print("Aanalyzing pca_dataset.txt")
    print("Our SVD implementation: ")
    pca_svd(path)
    print("Built-in PCA implementation: ")
    pca_built_in(path)

    # Task 1 Second Part
    img_path = cwd / "images" / "racoon.jpeg"
    print("Aanalyzing racoon.jpeg")
    pca_image(img_path)

    # Task 1 Third Part
    path = cwd/"datasets"/"data_DMAP_PCA_vadere.txt"
    print("Aanalyzing data_DMAP_PCA_vadere.txt")
    pca_trajectory(path)
