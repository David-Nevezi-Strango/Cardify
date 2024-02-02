import tensorflow as tf
import numpy as np
from main import rotateImg
from sklearn.model_selection import train_test_split
import os
import cv2
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def loadData(folder_path):
    images = []
    # for folder_path in folder_path_list:
    i = 0
    listdir = os.listdir(folder_path)
    random.shuffle(listdir)
    for image_name in listdir:
        if i == 20:
            break
        print(image_name)
        image_path = os.path.join(folder_path, image_name)

        # Load the image and preprocess it if needed
        image = cv2.imread(image_path)
        # apply rotation
        image = rotateImg(image)
        # need resizing to standard
        image = cv2.resize(image, (224,224))
        # Append the image and label to the lists
        images.append(image)
        i+=1

    return np.array(images)

def trainModel(path_pos_img, path_neg_img):
    #get the training data
    pos_data = loadData(path_pos_img)
    pos_label = np.ones(pos_data.shape[0])
    neg_data = loadData(path_neg_img)
    neg_label = np.zeros(neg_data.shape[0])

    #combine the negative and positive training data
    X = np.concatenate([pos_data, neg_data])
    y = np.concatenate([pos_label, neg_label])
    random_seed = 42
    np.random.seed(random_seed)

    # Shuffle the data
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_seed)
    #cast to tensorflow Dataset object type
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 8
    SHUFFLE_BUFFER_SIZE = 100

    #shuffle more and create batches
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # NN architecture
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])
    # IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

    model.fit(train_dataset,epochs=20)

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(X_test).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', y_test)

    model.save("mobilenetv2ID.keras")
    print("Training done and model saved!!")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    cr = classification_report(y_test, predictions)
    print(cr)

trainModel("id", "../img_celeba/imagenette2-320/train")