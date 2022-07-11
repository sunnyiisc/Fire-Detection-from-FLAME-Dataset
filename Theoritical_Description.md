# Fire-vs-NoFire Classification

Keras has an useful API which makes us easier to define the layers of our neural network. Here the input shape is $254\times254$ which is our image size and 3 represents color channel RGB.

- Conv2D() : Neural networks apply a filter to an input image to create a feature map that summarizes the presence of detected features in the input. In our case there are 32, 64, 128 and 128 filters or kernels in respective layers and the size of the filters are 3X3 with activation fucntions as relu.
- MaxPool2D() :Max pooling is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map.
- Flatten() : This method converts the multi-dimensional image data array to 1D array.

In addition to these a Rescaling layer is added to the model so that the image pixel values are re-scaled into 0-1 from 0-255. The complete CNN model used for the classification purpose as mentioned above is shown below:

![cnn_model](https://user-images.githubusercontent.com/47363228/178321808-f732cbb9-3493-4851-bd4e-75212e20447a.svg)

# Fire-Segmentation

A few changes are made to this network to accommodate the FLAME dataset and adapt it to the nature of this problem. The ReLU activation function is changed to Exponential Linear Unit (ELU) of each two- dimensional convolutional layer to obtain more accurate results. The ELU function has a negative outcome smaller than a constant value for the negative input values and it exhibits a smoother behavior than the ReLU function. The structure of the customized U-Net is shown below. The backbone of the U-Net consists of a sequence of up-convolutions and concatenation with high-resolution features from the contracting path. 

- The size of the input layer is $512 \times 512 \times 3$ designed to match the size of the inputs images and three RGB channels.
- For computational convenience, the RGB values (between 0 and 255) are scaled down by 255 to yield float values between 0 and 1.
- It follows the first contracting block including a two-dimensional fully convolutional layers with the ELU activation function, a dropout layer, another same fully convolutional layer, and a two-dimensional max pooling layer.
- This structure is repeated another three times to shape the left side of the U shape.
- Next, there are two two-dimensional fully connected layers with a dropout layer in between, the same structure of the left side is repeated for the right side of the U shape to have a symmetric structure for the up-convolution path in each block. Also, there exists a concatenation between the current block and the peer block from the contracting path.
- Since the pixel-wise segmentation is a binary classification problem, the last layer has the Sigmoid activation function.

The DCNN utilizes a dropout method to avoid the overfitting issue in the FLAME dataset analysis and realize a more efficient regularization noting the small number of ground truth data samples. The utilized loss function is the binary cross entropy. The Adam optimizer is used to find the optimal value of weights for the neurons.

![unet_model](https://user-images.githubusercontent.com/47363228/178321950-64891b3c-564a-4d1d-a6f4-7e6c328599f6.svg)


