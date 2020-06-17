"""
This file contains the implementation of Diffusion maps algorithm.

It contains several methods for different parts (1, 2, 3) of the Task.
The methods are named accordingly and the descriptions are given with them seperately.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from sklearn.datasets.samples_generator import make_swiss_roll
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


def generate_data_for_part_one(N):
    """
    This method is used to generate the data set for Part one of Task 2.

    Here we have to generate a time series data, where,
    t = (2*pi*k)/N+1,
    and the datapoints for set X are given by,
    x = (cos(t), sin(t)), at any k-th time moment.

    :param N: It is number of data points to be generated.
    :return: It returns a data frame of the data generated and the time.
    """
    for k in range(N):
        t = np.linspace(0, (2 * np.pi * k) / (N + 1), N)
        x = np.cos(t)
        y = np.sin(t)

    data_matrix = []
    for i in range(N):
        data_matrix.append([x[i], y[i]])

    df = pd.DataFrame(data_matrix, columns=['xcord', 'ycord'])

    """
    Here we plot the xcord on x-axis and ycord on y-axis of our data point X.
    We obtain a circular plot.
    """
    plt.figure(figsize=[8, 8])
    plt.plot(df['xcord'], df['ycord'])
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('periodic data set')
    plt.show()
    return df, t


def generate_data_for_swiss_roll(N, noise):
    """
    This method is used to generate the data set for Part two of Task 2. This data set
    is called Swiss Roll dataset.

    Here, we use built-in method make_swiss_roll() from sklearn library.

    :param N: It is number of data points to be generated.
    :param noise: It is the noise (if any) for the data
    :return: It returns a data frame containing the data for swiss roll data set.
    """
    # Generating the data points for swiss roll data set and put them in data frame
    data, data_color = make_swiss_roll(N, noise)
    idx_plot = np.random.permutation(N)[0:N]
    df = pd.DataFrame(data, columns=['xcord', 'ycord', 'zcord'])

    """
    Now we plot the swiss roll data set on a 3-d plot with columns of data frame - xcord, ycord, zcord 
    on x, y and z axes respectively.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[idx_plot, 0], data[idx_plot, 1], data[idx_plot, 2],
               c=data_color[idx_plot],
               cmap=plt.cm.Spectral)
    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.set_zlabel("z")
    ax.set_title("Swiss-Roll data manifold");
    ax.view_init(10, 70)
    plt.show()
    return df


def diffusion_maps(L, df):
    """
    This method implements all the steps of the diffusion algorithm as given in the sheet.

    :param L: It is the number of eigen vectors required.
    :param df: If is the data frame that we get from the methods where we generate the data.
    :return: It returns the final 'L' eigen values and eignen vectors and also the distance matrix to reconstruct the
             topology on a 2-D plot.
    """

    """
    The following block of code implements the formation of ambient kernel.
    """
    dist_matrix = pd.DataFrame(distance_matrix(df.values, df.values))
    epsilon = 0.05 * np.max(np.amax(dist_matrix.values))
    W_kernel_matrix = np.exp(-np.square(dist_matrix.values) / epsilon)

    """
    The following block of code implements the normalizing steps of the ambient kernel
    """
    P_diag_norm = np.diag(np.sum(W_kernel_matrix,
                                 axis=1))
    K_kernel_matrix = np.linalg.inv(P_diag_norm) @ W_kernel_matrix @ np.linalg.inv(P_diag_norm)
    Q_diag_norm = np.diag(np.sum(K_kernel_matrix,
                                 axis=1))
    Q_half = np.power(Q_diag_norm, 0.5)
    Q_inv_half = np.linalg.inv(Q_half)
    T_symm_matrx = Q_inv_half @ K_kernel_matrix @ Q_inv_half

    """
    The following last block calculates the 'L' largest eigen values and the corresponding eigen vectors.
    """
    eig_val, eig_vec = np.linalg.eigh(T_symm_matrx)
    eig_val_needed = np.flip(eig_val[-L:])
    A = np.flip(eig_vec, axis=1)
    eig_vec_needed = A[:, :L]
    T_eigen_values_epsilon = np.sqrt(np.power(eig_val_needed, 1 / epsilon))
    phi_eigen_vectors = Q_inv_half @ eig_vec_needed

    return phi_eigen_vectors, T_eigen_values_epsilon, dist_matrix


# Part 1
def part_one_diff_maps(N, L):
    """
    This method implements the plotting of first five eigen vectors against time.
    The time is on x-axis and the eigen vectors are on y-axix.

    :param N: It is the number of data points.
    :param L: It is the number of eigen vectors needed.
    """

    """
    Here we get the data generated and store in a data frame, and then run the diffusion algorithm on this data.
    """
    df, t = generate_data_for_part_one(N)
    L_eigen_vectors, L_eigen_values, _ = diffusion_maps(L, df)

    """
    We now plot each eigen vector in y-axis to time on x-axis.
    """
    for each in range(L):
        plt.plot(t.reshape((1000, 1)), L_eigen_vectors[:, each],
                 label='Eigen Value is ' + str(L_eigen_values[each]))
        plt.xlabel('time')
        plt.ylabel('eigen vector - ' + str(each))
        plt.legend()
        plt.show()


def fourier_analysis(N):
    """
    This method implements the fourier analysis for the data time series generated.

    :param N: It is the number of datapoints generated.
    """
    for k in range(N):
        t = np.linspace(0, (2 * np.pi * k) / (N + 1), N)
        x = np.cos(t)
        y = np.sin(t)

    """
    Normalize amplitude of both x and y.
    Here we use fft package of numpy to perform the fourier analysis.
    """
    fourierTransform = np.fft.fft(x+y)
    freq = np.fft.fftfreq(t.shape[-1])
    """
    Here we plot the final 'signal' with the normalized amplitude.
    """
    plt.title('Fourier transform depicting the frequency components')
    plt.plot(freq, fourierTransform.real, freq, fourierTransform.imag)
    plt.xlabel('Frequency')
    plt.ylabel('Fourier Transformed value')
    plt.show()


# Part 2
def part_two_diff_maps(N, noise, L):
    """
    This method implements the plotting of first 10 eigen functions of Laplace-Beltrami operator
    against the first eigen vector for swiss-roll dataset.

    :param N: It is the number of data points in swiss-roll dataset.
    :param noise: It is the noise of data (0 in our case).
    :param L: It is the number of required eigen funtions
    """

    """
    Here we generate the swiss-roll dataset in a data frame and run the diffusion algorithm on this data. 
    """
    df = generate_data_for_swiss_roll(N, noise)
    L_eigen_vectors, L_eigen_values, _ = diffusion_maps(L, df)

    # Reconstruct the data along the eigen vectors
    new_data = df.values @ L_eigen_vectors[:, :3].T

    """
    For clarity, we plot the reconstructed data along the eigen vectors as a scatter plot. 
    We take 1st eigen function on x-axis and the others on y-axis.
    """
    for each in range(L):
        plt.scatter(new_data[0], new_data[each], c=new_data[0],
                    cmap=plt.cm.Spectral)
        plt.xlabel('eigen vector - 1')
        plt.ylabel('eigen vector - ' + str(each))
        plt.show()
        plt.show()


def part_two_pca_swiss_roll(N, noise):
    """
    Thi method performs the Pricipal Component Analysis for the swiss-roll dataset and
    plots first 2 and first 3 principal components respectively.

    :param N: It is the number of data points.
    :param noise: It is the noise of data (0 in our case).
    """

    """
    Here we generate the swiss-roll data set from our method above and perfom PCA on this data.
    """
    df = generate_data_for_swiss_roll(N, noise)
    centered_dataset = (df - df.mean())
    pca = PCA()
    principalComponents = pca.fit_transform(centered_dataset)
    print(pca.explained_variance_ratio_)

    """
    Here we plot the swiss roll data set along two Principal components
    """
    ax = plt.axes(projection='3d')
    ax.scatter(principalComponents[:, 0], principalComponents[:, 1],
               cmap=plt.cm.Spectral)
    plt.title("Swiss roll Dataset plotted against 2 principal components")
    plt.show()

    """
    Here we plot swiss roll data set along three Principal components
    """
    ax = plt.axes(projection='3d')
    ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2],
               cmap=plt.cm.Spectral)
    plt.title("Swiss roll Dataset plotted against 3 principal components")
    plt.show()


# Part 3
def part_three_trajectory(path, L):
    """
    This method runs the iffusion algorithm on the trajectory data set provided.

    :param path: It is the path where the data set is stored
    :param L: It is the number of eigen functions needed.
    """

    """
    Here, we read the data from the .txt file provided which is stored in datasets directory here.
    We then run the diffusion algorithm on this data set.
    """
    ped_path = pd.read_csv(path, sep=' ').to_numpy()
    df = pd.DataFrame(ped_path)
    L_eigen_vectors, L_eigen_values, _ = diffusion_maps(L, df)
    ax = plt.axes(projection='3d')

    """
    Here we plot 3 eigen functions on a 3-d plot. This gives a clear visualisation of the trajectory.
    """
    ax.scatter(L_eigen_vectors[:, 0], L_eigen_vectors[:, 1], L_eigen_vectors[:, 2],
               cmap=plt.cm.Spectral)
    ax.set_title('Trajectories plotted along 3 eigen functions')
    plt.show()

    """
    Here we plot each eigen function on y-axix against 1st eigen function on x-axix.
    """
    for each in range(L):
        plt.plot(L_eigen_vectors[:, 1], L_eigen_vectors[:, each])
        plt.title('eigen vector - 1 vs eigen vector - '+str(each))
        plt.xlabel('eigen vector - 1')
        plt.ylabel('eigen vector - ' + str(each))
        plt.show()


"""
This is the main methods which calls the other methods to run the implementation.
"""
if __name__ == '__main__':
    part_one_diff_maps(1000, 5)
    fourier_analysis(1000)
    part_two_diff_maps(5000, 0.0, 10)
    part_two_pca_swiss_roll(5000, 0.0)
    cwd = Path.cwd()
    path = cwd/"datasets"/"data_DMAP_PCA_vadere.txt"
    part_three_trajectory(path, 6)
