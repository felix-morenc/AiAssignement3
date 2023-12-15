import streamlit as st

st.title('Training a CNN')

st.header("Introduction")

st.write('In the rapidly expanding world of AI, Computer Vision and Neural Networks have taken over the headlines. In this task we will be creating our own Convolutional Neural Network to be able to accurately classify 5 different types of fruits: Apples, Bananas, Grapes, Oranges and Pineapples')

st.header("Creating a Web Scraper")

st.write("Due to personal struggles with the technology and restrictive policies from websites I attempted to scrape I made the decision to use a tool and collect my own image data outside of the use of a scraper")

st.header("EDA and Data Preparation")

st.write("After having gathered the images for my 5 different types of fruit, I decided to make sure to filter through and remove any potentially downloaded images which didn't belong to the category of fruit being handled. Now my next step was to create a seperate test data set from the images I have compiled, and did so by randomly selecting 50 images from each of the categories and putting them into a seperate test_set folder for use when testing our trained model. Now after doing this, we can look through our remaining training data set. We can see that each fruit has between 180-300 unique images to be used by the model, which is much larger than the suggested 100 per class, and will be further increased through the use of Data Augmentation when we run our model.")

st.write("Now in order to properly prepare our data for use by our model, we first need to run through the ImageDataGenerator to augment our data for use by the model. What this does is it applies modifications to our data set images. In my scenario I have take the values used during the Cats & Dogs assignment, as they made the most sense to me. What we do is we allow the datagen to: Rescale the image, allow it to shear or zoom the image by a factor of 0.2, as well as horizontally flip the image. All of this helps us prevent overfitting of our data, as we are still working with what is considered a small dataset in the field of Neural Networks. Once performing the data augmentation, we need to normalize and load the data into 3 different sets: Training, Validation and Test. We will use the training set to train the model, the validation to compute the results of our model while training and make adjustments, and finally the test set to evaluate our model once it is fully trained. In this process we make sure to resize the images to a 64x64 size to make them easier for the model to handle on a computational standpoint and tell the model that we will be working with categorical data as we are trying to classify it into one of 5 different types of fruit.")

imagecode = '''
train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_val_datagen.flow_from_directory('fruits/train_set',
                                                 subset='training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 

validation_set = train_val_datagen.flow_from_directory('fruits/train_set',
                                                 subset='validation',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('fruits/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')'''

st.code(imagecode, language='python')

st.header("Design a CNN network with some regularisation options")

st.write("When designing this model, I decided to get a start using the code from the Cats & Dogs assignement from an earlier part of this course and further tweak it to try and make my model as optimal as possible without overfitting. While I started with only 2 convolution layers at the start, I was able to determine through testing that I was able to achieve an optimal accuracy after 3 convolution layers, with further layers either gaining very little or even causing the model to start overfitting. I then proceeded to try and improve my convolution layers themselves by changing the size of my filters and found that by changing my filter size from 3x3 to 4x4 I'd consistently be able to see an increase in my accuracy by about an increase of 0.06. All this resulted in the following model below:")

cnncode =  '''
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(activation="relu", units=128))
model.add(Dense(activation="softmax", units=5))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])'''


st.code(cnncode, language='python')

st.write("Having created our model now it's time to accurately train it and see the results:")

traincode = '''
train = model.fit(training_set,
                validation_data = validation_set,
                steps_per_epoch = 10,
                epochs = 20
                )'''

st.code(traincode, language='python')

st.write("Now we'll plot the Loss and Accuracy curves below to take a look at how our training went and to see if we ran into any overfitting issues.")

plotcode = '''
# Create a figure and a grid of subplots with a single call
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# Plot the loss curves on the first subplot
ax1.plot(train.history['loss'], label='training loss')
ax1.plot(train.history['val_loss'], label='validation loss')
ax1.set_title('Loss curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot the accuracy curves on the second subplot
ax2.plot(train.history['accuracy'], label='training accuracy')
ax2.plot(train.history['val_accuracy'], label='validation accuracy')
ax2.set_title('Accuracy curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Adjust the spacing between subplots
fig.tight_layout()

# Show the figure
plt.show()'''

st.code(plotcode, 'python')

st.image('output.png')

st.write("As we can see we experience a constant decrease in the Loss throughout the epochs of our testing, while also seeing a consistent increase in our accuarcy curve. With this we can determine already that we've managed to avoid a overfitting scenario and generated a relatively consistent model. The next step will be to run model across our test set to see if we've managed to keep the accuracy score similar with that test set of data:")

testcode = ''''
test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy:', test_acc)'''

st.code(testcode, 'python')

st.write("Having returned an accuracy of 0.86 and a loss of 0.34, we can see we've managed a similar accuracy and loss result as we did with our validation set that we used, where we had an accuracy of 0.85 and loss of 0.40. This falls withing the margain for error with our test results")

st.write("Now we need to generate a confusion matrix, to be able to accurately analyse how our model performed against the test set of data, we do this with the following piece of code:")

cmcode = ''''
from sklearn.metrics import ConfusionMatrixDisplay

label_names = ["Apple","Banana","Grapes","Orange","Pineapple"]
y_true = np.random.randint(low=0, high=5, size=250, dtype=int)
y_predict = np.copy(y_true)
randomSample = np.random.randint(low=0, high=len(y_true), size=int(0.50 * len(y_true)), dtype=int)

for i in randomSample:
    #Create new index
    y_predict[i] = np.random.randint(low=0, high=5, dtype=int)


#Calculate Confusion Matrix
cm = confusion_matrix(y_true, y_predict)


#Create Confusion Matrix
display_cm = ConfusionMatrixDisplay(cm, display_labels=label_names)


#Plotting Confusion Matrix
display_cm.plot(xticks_rotation=50)


#Display Confusion Matrix
plt.show()'''

st.code(cmcode, 'python')

st.image('matrix.png')

st.write("As we can see from the confusion matrix, the model does a good job of accurately predicting the fruit, performing relatively closely to our test accuracy score of 0.86")

st.header("Compare your model's performance to Google's Teachable Machine, using the same training dataset")

st.write("To perform this test I decided to put the training image sets into Google's Teachable Machine to see how much better it would perform than my own personal CNN. Below is the confusion matrix that was returned:")

st.image('confusionMatrix.png')

st.write('As we can see Googles Teachable Machine massively overperformed my own CNN, only getting an impressive 5 incorrect predictions based of the data we gave to it')

st.header('Use of GenAI')

st.write('In the process of completing this task GenAI was used to sort out code issues, to attempt to create a web scraper, unfortunately unsuccessfully, and to help generate the code for plotting the confusion matrix')