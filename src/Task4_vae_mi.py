import h5py
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt



def get_mnist_data():
    """
    Downloads the mnist dataset from keras
    :return: train_images, test_images -> training and test data-set in a 2-Dimensional shape
    """
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    # Normalise the data
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0

    # Reshape the matrices to be 2-Dimensional
    train_x = train_x.reshape(len(train_x), np.prod(train_x.shape[1:]))
    test_x = test_x.reshape(len(test_x), np.prod(test_x.shape[1:]))

    # print(train_x.shape)
    # print(test_x.shape)

    return train_x, test_x, train_y, test_y


def encoder_model(prior, original_dim=784, intermediate_dim=256, latent_dim=2, activation='relu'):
    """
    Creates an encoder model with all its layers
    :param prior: For applying KL-Divergence regularizer to the model
    :param intermediate_dim: dimension of the hidden layer, default 256
    :param latent_dim: dimension of latent space, default 2
    :return: encoder model of type tf.keras.Model
    """

    original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(original_inputs)
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(intermediate_dim, activation=activation)(x)
    z = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim))(x)
    z = tfp.layers.MultivariateNormalTriL(latent_dim,
                                          activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior))(z)
    return tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')


def decoder_model(original_dim=784, intermediate_dim=256, latent_dim=2, activation='relu'):
    """
    Create a decoder model with all its layers
    :param intermediate_dim: dimension of the hidden layer, default 256
    :param original_dim: original dimension of of the image, default 784
    :return:
    """

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(intermediate_dim, activation=activation)(x)
    outputs = tf.keras.layers.Dense(original_dim)(x)
    outputs = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfp.distributions.MultivariateNormalDiag(loc=t),
        convert_to_tensor_fn=tfp.distributions.Distribution.mean)(outputs)
    return tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')


def plot_digits_from_prior(prior, decoder, n=15, cols=5):
    """
    Plots n digits sampled from prior
    :param prior: Distribution from which you want to sample digits
    :param n: number of samples, default 15
    :param cols: number of digits per row
    :return: None.
    """

    sample = prior.sample(n)
    fig = plt.figure()
    plot_n = 1
    rows = n // cols + bool(n % cols)

    for i in sample:
        sam = np.array([i.numpy()])
        gen = decoder.predict(sam)[0].reshape(28, 28)
        ax = fig.add_subplot(rows, cols, plot_n)
        ax.axis('off')
        ax.imshow(gen, cmap='Greys_r')
        plot_n += 1
    plt.suptitle(str(n) + ' generated digits from prior')
    plt.show()


def plot_reconstructed_original_digits(encoder, decoder, data):
    """
    Plots n digits sampled from prior
    :param prior: Distribution from which you want to sample digits
    :param cols: number of digits per row
    :return: None.
    """

    fig = plt.figure(figsize=(15.0, 5.0))

    for i in range(data.shape[0]):
        # Plot original image

        ax = fig.add_subplot(2, data.shape[0], i + 1)
        ax.axis('off')
        # ax.suptite('Original Image')
        ax.imshow(data[i].reshape(28, 28), cmap='Greys_r')

        # Plot reconstructed image

        reconstructed = decoder.predict(encoder.predict(np.array([data[i]])))
        ax = fig.add_subplot(2, data.shape[0], i + 1 + data.shape[0])
        ax.axis('off')
        # ax.suptitle('Reconstructed Image')
        ax.imshow(reconstructed.reshape(28, 28), cmap='Greys_r')

    plt.show()


def plot_latent_repr(encoder, data, labels, batch_size=128):
    """
    Plot the latent representation
    :param encoder: Encoder model which outputs the latent values, should be of type tf.keras.Model
    :param data: Input data for encoder, should be of shape (original_dim,)
    :param batch_size: Batch size of type int
    :return: None. Plots the latent representation
    """

    test_encoded = encoder.predict(data, batch_size=batch_size)
    plt.figure(figsize=(10, 8))
    plt.scatter(test_encoded[:, 0], test_encoded[:, 1], c=labels, cmap='rainbow')
    plt.axis('off')
    plt.title('Latent Representation')
    plt.colorbar()
    plt.show()


def plot_loss(history, epochs, label='Loss'):
    """
    Plots loss vs epoch graph for a given model
    :param model: Model for plotting it loss. Should be of type tf.keras.Model
    :param epochs: Number of epochs trained for. Should be of type int
    :return: None, plots loss vs epoch curve
    """

    loss = history.history['loss']
    plt.figure()
    plt.plot(range(1, epochs + 1), loss, label=label)
    plt.title(label)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def encoder_predict(encoder, data):
    """
    Outputs the latent space values
    :param encoder: encoder model of type tf.keras.Model
    :param data: 2-d input data
    :return: a tensor of shape (none, latent_dim) which can directly be fed to decoder
    """
    return encoder.predict(data)


def decoder_predict(decoder, latent_data):
    """
    Outputs reconstructed values from latent space inputs
    :param decoder: decoder model of type tf.keras.Model
    :param latent_data: Must be of shape(batch_size, latent_dim)
    :return: decoder output of shape original dimension which was passed to the model
    """
    return decoder.predict(latent_data)


def get_models(prior, original_dim, intermediate_dim, latent_dim, activation):
    """
    Initialise models
    :param prior: prior probability distribution of type tfp.distributions.Distribution
    :param original_dim: original dimension of input data to the encoder.
    :param intermediate_dim: dimensions of intermediate layer, of type int
    :param latent_dim: latent dimension of type int
    :return: Encoder, decoder and vae models
    """
    encoder = encoder_model(prior, original_dim, intermediate_dim, latent_dim, activation)
    decoder = decoder_model(original_dim, intermediate_dim, latent_dim, activation)
    return encoder, decoder, tf.keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))


def plot_data(data_train, data_test):
    """
    Plots a scatter plot of the training data
    :param data: 2-dimensional data
    :return: None. plots a scatter plot
    """
    plt.figure()
    plt.scatter(data_train[:, 0], data_train[:, 1], label='train')
    plt.scatter(data_test[:, 0], data_test[:, 1], label='test')
    plt.legend()
    plt.title('data')
    plt.show()


def plot_reconstructed(encoder, decoder, data):
    """
    Plot reconstructed data
    :param encoder: encoder model of type tf.keras.Model
    :param decoder: decoder model of type tf.keras.Model
    :param data: input data for the encoder model, shape (n, original_dim) where n is number of samples
    :return: reconstructed - x-y coords of reconstructed data, plots reconstructed data
    """
    reconstructed = decoder.predict(encoder.predict(data))
    plt.figure()
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1])
    plt.title('Reconstructed')
    plt.show()
    return reconstructed


def plot_reconstructed_original_digits(encoder, decoder, data):
    """
    Plots n digits sampled from prior
    :param prior: Distribution from which you want to sample digits
    :param cols: number of digits per row
    :return: None.
    """

    fig = plt.figure(figsize=(15.0, 5.0))

    for i in range(data.shape[0]):
        # Plot original image
        ax = fig.add_subplot(2, data.shape[0], i + 1)
        ax.axis('off')
        ax.imshow(data[i].reshape(28, 28), cmap='Greys_r')

        # Plot reconstructed image
        reconstructed = decoder.predict(encoder.predict(np.array([data[i]])))
        ax = fig.add_subplot(2, data.shape[0], i + 1 + data.shape[0])
        ax.axis('off')
        ax.imshow(reconstructed.reshape(28, 28), cmap='Greys_r')

    plt.show()


def plot_generated_data(prior, decoder, n=15, data=None, reconstructed=None):
    """
    Plots n digits sampled from prior
    :param prior: Distribution from which you want to sample digits
    :param n: number of samples, default 15
    :param cols: number of digits per row
    :return: reconstructed - [x, y] data points
    """

    sample = prior.sample(n)
    plt.figure()
    generated = decoder.predict(sample)
    if data is not None:
        plt.scatter(data[:, 0], data[:, 1], label='Train Data', s=17)
    if reconstructed is not None:
        plt.scatter(reconstructed[:, 0], reconstructed[:, 1], label='Reconstructed Data', s=15)
    plt.scatter(generated[:, 0], generated[:, 1], label='Generated Data', s=9)
    # plt.axis('off')
    plt.title(str(n) + ' generated samples from prior')
    plt.legend()
    plt.show()
    return reconstructed


def find_critical_point(decoder, prior):
    """
    Find out the critical point as per the specs given in the task
    :param decoder: decoder model of type tf.keras.Model
    :param prior: prior probability distribution of type tfp.distributions
    :return: gen, gen_pop, box_pop
    """
    gen = []
    gen_pop = []
    box_pop = 0
    while box_pop < 100:
        temp = decoder.predict(prior.sample(1))
        gen.append(temp[0])
        if 130 <= temp[0][0] <= 150 and 50 <= temp[0][1] <= 70:
            box_pop += 1
            gen_pop.append(len(gen))
            # print(box_pop)
    return gen, gen_pop, box_pop


def plot_critical_pop(data):
    """
    Plot the generated population at critical point
    :param data: 2-d data
    :return: none, plot the data
    """
    plt.figure()
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], s=9, c='green')
    plt.title(str(len(gen)) + ' generated samples from prior')
    plt.show()

if __name__ == '__main__':
    # Load the data
    train = np.load('FireEvac_train_set.npy')
    # train_val = train[:200]
    # train = train[200:]
    test = np.load('FireEvac_test_set.npy')
    plot_data(train, test)

    # Hyper-Parameters
    epochs = 12500
    learning_rate = 1e-3
    batch_size = 64
    original_dim = 2
    intermediate_dim = 128
    latent_dim = 32
    activation = 'softplus'
    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_dim),
                                                     scale_diag=tf.ones(latent_dim))  # Do not change this!

    try:

        # Initialise the models
        encoder, decoder, vae = get_models(prior, original_dim, intermediate_dim, latent_dim, activation)

        # Train the model
        vae.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                    loss=lambda y_true, y_pred: -y_pred.log_prob(y_true))
        # history = vae.fit(train, train, epochs=epochs, batch_size=batch_size)

        # Check if there is a trained model before
        path = "./../models/vae_mi.h5"
        file = h5py.File(path, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)][:])
        vae.set_weights(weight)

        path = "./../models/encoder_mi.h5"
        file = h5py.File(path, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)][:])
        encoder.set_weights(weight)

        path = "./../models/decoder_mi.h5"
        file = h5py.File(path, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)][:])
        decoder.set_weights(weight)

        # Plot reconstructed data
        reconstructed = plot_reconstructed(encoder, decoder, test)

        # Generate 1000 points
        generated = plot_generated_data(prior, decoder, n=1000, data=train, reconstructed=reconstructed)

        # Find-out the critical point
        gen, gen_pop, box_pop = find_critical_point(decoder, prior)

        # Print the critical point
        print('Critical point = ', len(gen))

        # Plot the generated population at critical point
        plot_critical_pop(gen)

        plt.figure()
        plt.plot(gen_pop, range(0, box_pop))
        plt.title("Total_population vs Box_population")
        plt.show()
        # np.save('generated_data', generated)

    except:

        # Initialise the models
        encoder, decoder, vae = get_models(prior, original_dim, intermediate_dim, latent_dim, activation)

        # Train the model
        vae.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                    loss=lambda y_true, y_pred: -y_pred.log_prob(y_true))
        history = vae.fit(train, train, epochs=epochs, batch_size=batch_size)

        # Plot loss curve
        plot_loss(history, epochs)

        # Plot reconstructed data
        plot_reconstructed(encoder, decoder, test)

        # Plot generated samples
        reconstructed = plot_generated_data(prior, decoder, n=1000)

        # Save model
        file = h5py.File('./../models/vae_mi.h5', 'w')
        weight = vae.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()
        file = h5py.File('./../models/decoder_mi.h5', 'w')
        weight = decoder.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()
        file = h5py.File('./../models/encoder_mi.h5', 'w')
        weight = encoder.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()
