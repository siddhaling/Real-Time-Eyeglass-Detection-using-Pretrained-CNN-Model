# Eyeglass Detection
Real-time Eyeglass Detection framework based on deep features present in facial or ocular images, which serve as a prime factor in forensics analysis, authentication systems and many more.
## Developed by Ritik Jain
# Getting Started
* [Training and Optimizing](Training_and_Optimizing.ipynb) extracts the features using InceptionV3 from custom dataset. The feature extraction and training process is fine-tuned by Optimizing different model hyper-parameters.
* [Haar-cascade](haarcascade_face.xml) defines the haar features for extraction of face during real-time.
* [Saved model](https://drive.google.com/file/d/1y__WYYk1SxB6dhYn8kEXdzDdKm9UJMOa/view?usp=sharing) includes the saved model weights of fine-tuned model.
* [Runnable framework](output.py) captures frames in real-time, extract faces and detect the presence of eyeglass for each face.

## Requirements
Python 3+, TensorFlow, Keras and other common packages listed in `requirements.txt`.

## Installation and Execution
1. Install dependencies (Anaconda also has to be installed independently and its bin folder must be included to the PATH environment variable)
   ```bash
   pip install -r requirements.txt
   ```
2. Clone this repository
3. Open anaconda prompt and change location to the cloned repository
4. Run the real-time framework by using the command
    ```bash
    python output.py
    ```

# Data Source
A custom dataset of non-standard facial images was used to refactor the pre-trained model and update the weights on top layers to make the model robust in making prediction for real-life data. The dataset used for modelling contains a total of 46,372 images belonging to two classes namely eyeglasses present (label-0) and eyeglasses absent (label-1). Three sources of data from which the data was collected are mentioned below.
* The first part of dataset is custom built wherein, facial image data was collected from few families and friends by requesting them to click and send their pictures with and without eyeglasses in different illuminations and orientations. This part of the dataset involves a total of 4062 images from 20 subjects.
* The database released by the authors of “An Indian Facial Database Highlighting the spectacles problem contains 123,213 images of 58 subjects with and without eyeglasses in multiple sessions. A random sub-sample of about 30% was selected in equal proportions from each subject to give a total of 37,388 images belonging to both output classes in a balanced ratio.
* The Kaggle dataset of Glasses or No Glasses containing 4,922 images was used in the third part of our dataset. This dataset contains high-quality pictures and thus bringing diversity to our dataset.

# Method
The input images need to be pre-processed before being used for training. OpenCV supports object recognition using Haar features which can recognise faces in an image. Using this approach, a new dataset is thus generated which is then preprocessed based on the requirements of the transfer learning model i.e., resizing to 299,299 and normalising each pixel.
An InceptionV3 architecture based transfer learning model was used to allow a fine model training process.  Next step involves training of the model consisting of 262,146 trainable parameters over a total of 22,064,930 parameters present in the proposed model. Multiple models were trained on the batch image dataset while varying the values of different hyper parameters to find the optimised version of proposed model and hence fine tuning the model.
The following hyper-parameters of the model were optimised by selecting the best performing value for each parameter:
1.	Train-Test Split Ratio
2.	Optimizer Function
3.	Learning Rate
4.	Activation Function (for fully connected layer)
For real time detection framework, the model trained on the fine-tuned hyper parameters is used and loaded through saved model in .h5 file. This loaded model is used to make predictions on frame captured in real-time by OpenCV.

# Result
The model was trained on 80% of the pre-processed data using Adagrad optimizer for the categorical cross entropy loss function using an initial learning rate value of 0.001 while using softmax activation for our final dense classification layer to get results as mentioned in the table below. The model was evaluated on the left out 20% of the above data termed as test data. To further test the robustness and performance of our model, another small custom-built dataset termed as real test data was used which includes subjects different from those in the main dataset. Two additional independent test datasets namely ORL and Sunglasses datasets were collected from published resources to further test the generalization capability of our model.
Performance metrics (accuracy) for optimised model on different datasets are mentioned below

* Train Data (80%)- 99.28
* Test Data (20%)- 99.95
* Real Test Data- 100.00
* ORL Test Data- 94.38
* Sunglasses Test Data- 98.20


# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passionate Researcher, Focus on Deep Learning and its applications,\
dr.siddhaling@gmail.com
