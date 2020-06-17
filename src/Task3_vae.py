import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


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


def encoder_model(prior, intermediate_dim=256, latent_dim=2):
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
    z = tf.keras.Sequential([
        tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim)),
        tfp.layers.MultivariateNormalTriL(latent_dim),
        tfp.layers.KLDivergenceAddLoss(prior)
    ])(x)
    return tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')


def decoder_model(intermediate_dim, original_dim=784):
    """
    Create a decoder model with all its layers
    :param intermediate_dim: dimension of the hidden layer, default 256
    :param original_dim: original dimension of of the image, default 784
    :return:
    """

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(x)
    outputs = tf.keras.layers.Dense(original_dim)(x)
    outputs = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfp.distributions.MultivariateNormalDiag(loc=t, scale_diag=tf.exp(t)),
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

    print(history.history.keys())
    loss = history.history['loss']
    test_loss = history.history['val_loss']
    plt.figure()
    # plt.plot(range(1, epochs + 1), loss, label='Train' + label)
    plt.plot(range(1, epochs + 1), test_loss, label='Test' + label)
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


def get_models(prior, original_dim, intermediate_dim, latent_dim):
    """
    Initialise models
    :param prior: prior probability distribution of type tfp.distributions.Distribution
    :param original_dim: original dimension of input data to the encoder.
    :param intermediate_dim: dimensions of intermediate layer, of type int
    :param latent_dim: latent dimension of type int
    :return: Encoder, decoder and vae models
    """
    encoder = encoder_model(prior, intermediate_dim, latent_dim)
    decoder = decoder_model(intermediate_dim, original_dim)
    return encoder, decoder, tf.keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))


if __name__ == '__main__':
    """
    # 2 Latent dimensions
    # Hyper-parameters
    
    You can comment the following block of code until next multiline comment if you want to train
    on just 1 model or you can chance hyper parameters as you like.
    """

    epochs = 120
    learning_rate = 1e-3
    batch_size = 128
    original_dim = 784
    intermediate_dim = 256
    latent_dim = 2
    prior = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(latent_dim))  # Do not change this!

    # Get the data
    train, test, train_y, test_y = get_mnist_data()

    # Initialise the models
    encoder, decoder, vae = get_models(prior, original_dim, intermediate_dim, latent_dim)

    # Train the model
    vae.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                loss=lambda y_true, y_pred: -y_pred.log_prob(y_true))
    history = vae.fit(train, train, epochs=epochs, batch_size=batch_size, validation_data=(test, test))

    # Plot 15 digits samples from prior
    plot_digits_from_prior(prior, decoder, n=15)

    # Plot latent representation of test data
    plot_latent_repr(encoder, test, test_y, batch_size=batch_size)

    # Plot some original and reconstructed images
    plot_reconstructed_original_digits(encoder, decoder,
                                       test[np.random.choice(len(test), size=15, replace=False)])

    # Plot loss vs epoch
    plot_loss(history, epochs)


    """
    # Train the model with 32 latent dimensions
    # Hyper-parameters
    """
    epochs = 100
    learning_rate = 1e-3
    batch_size = 128
    original_dim = 784
    intermediate_dim = 256
    latent_dim = 32
    prior = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(latent_dim))

    # Get the data
    train, test, train_y, test_y = get_mnist_data()

    # Initialise the models
    encoder, decoder, vae = get_models(prior, original_dim, intermediate_dim, latent_dim)

    # Train the model
    vae.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                loss=lambda y_true, y_pred: -y_pred.log_prob(y_true))
    history = vae.fit(train, train, epochs=epochs, batch_size=batch_size, validation_data=(test, test))

    # Plot 15 digits samples from prior
    plot_digits_from_prior(prior, decoder, n=15)

    # Plot some original and reconstructed images
    plot_reconstructed_original_digits(encoder, decoder,
                                       test[np.random.choice(len(test), size=15, replace=False)])

    # Plot loss vs epoch
    plot_loss(history, epochs)
