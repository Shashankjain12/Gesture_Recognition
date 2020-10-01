# Hand Gesture Recognition And Recommendation System 

This Repository is a project which involves recognition of Hand Gesture using Computer Vision.

This project aims to solve the problem for Deaf and Dumb People by using this technology they by just showing the gestures
Gestures can be converted into characters which can be used to form the words.

<img src="images/hand_signs.png" width="500" height="300"></img>

By using Computer Vision Technology I have used CNN which stands for Convolution Neural Network for recognizing and recommending appropriate words from the human hands gesture.

## Convolution Neural Network

CNN consists of Multiple Layers which can directly extract features from the Dataset in our case the dataset consists of
series of hand gestures namely from A-Z which consists of multiple folder each folder representing number of images about 
100-200 images per character. 

So, In General what CNN does is it extracts features from the images and the features are extracted by variety of layers
which is known to the network. It consists of several layers each layer have it's own function which basically reduces the
size of image by performing certain operations.

<img src="images/conv.jpg"></img>


While creation of CNN I have used many layers for performing these operations which are:-

### Convolution Layer

This is the most important layer used by any CNN based Network which by using convolution operation between image and convolution filter is used to reduce the information that is not required by the network so it removes all of the information which is not required for learning of details by the model.

### BatchNormalisation Layer

This layer is generally used to normalize that is to reduce the redundant information from the images which basically removes all of the information which can act as noise while creation of network which can highly affect the accuracy of the Network.

### MaxPooling Layer

This layer is used to extract certain information which is most useful in the images which can be used to extract the information which can be served the most important part while recognizing any of the images.

### Activation Layer

Activation Layer is generally as function which can be used as an average operation of all of the functions.
I have used RelU and Softmax Function as a activation layer in my Neural Network In which ReLU serves as the Rectifier function which involves giving positive values always for all positive values of x and then Softmax function is used to identify whether the output is in the form of integers. 

### Dense Layer 

This layer serves as Neural Network Fully connected layer which is used to pass input from all of the above layer to this dense function where we can set number of neurons and number of layers which can be used to serve the purpose of the Neural Network. 

While building this Neural Network I have used certain hyperparameters such as batch_size, epochs and the image size which can be used for training and testing as 28 * 28.

Here is the flow diagram Representing the whole process:-

<img src="images/Flow_diag.jpg" width="500" height="300"></img>

# Training Phase

Training Phase involves certain steps first is creation of data I have used my own custom dataset which is in the form of binary images which I will share later. This creation of binary images involved creation of images which consists of removal of background that is extraction of hand gestures from the images which involves background noise removal, binary image creation and then saving the images which are extracted from the frames.

Some of the training images which are generated after binary conversion are:-

<img src="images/train_a.jpg" width="100" height="100"></img>

<img src="images/train_w.jpg" width="100" height="100"></img>

# Testing Phase

After Training some amount of images that are created from the custom dataset are tested with the model which have been trained from the same size of 28 * 28 pixels images generated in our custom dataset thereby with the help of keras a mathematical computing library after creation of several layers i am able to get the accuracy of about 91% trained from the batch of 64 and epochs of about 100 to train our machine learning model.

<img src="images/test.jpg"></img>

# Hand Gesture Recognition + Recommendation

I have tried to merge recognition and Recommendation system both altogether in one system through which my model can correctly predict the characters as well then giving them correct recommendation of words so for the genral workflow i have used textgenrnn for creation of Recommendation system through which I have used an inbuilt dataset to train that generator RNN network for recommending a certain 3 character word from the character once it's been identified. So, as part of it the full workflow that I have been able to create is recognition of character with 90% accuracy then recommending a word from a character to generate a word from it which can be used by deaf and dumb to use my network in order to train the set of images.

<img src="images/hand_img_1.png" width="500" height="300"></img>

<img src="images/hand_img_2.png" width="500" height="300"></img>

## Recommendation

For Recommendation from set of words to the characters is done using [TextGenRNN library](https://github.com/minimaxir/textgenrnn).
textgenrnn is a Python 3 module on top of Keras/TensorFlow for creating char-rnns, with many cool features:

1. A modern neural network architecture which utilizes new techniques as attention-weighting and skip-embedding to accelerate training and improve model quality.
2. Train on and generate text at either the character-level or word-level.
3. Configure RNN size, the number of RNN layers, and whether to use bidirectional RNNs.
4. Train on any generic input text file, including large files.
5. Train models on a GPU and then use them to generate text with a CPU.
6. Utilize a powerful CuDNN implementation of RNNs when trained on the GPU, which massively speeds up training time as opposed to typical LSTM implementations.
7. Train the model using contextual labels, allowing it to learn faster and produce better results in some cases.

# Result 

Here is the resulting video that I have created for my system which is able to correctly identify the words and then generate the Words from the model.


![](https://media.giphy.com/media/els7zsKB6GTTPqOfw9/giphy.gif)

## Thanks For Watching :smiley:
