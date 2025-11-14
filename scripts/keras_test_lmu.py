
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os
import nengo
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense
import keras
import keras_lmu
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve

sys.path.insert(0,'../networks')
from ldn import LDN

sys.path.insert(0,'../')
from utilities import generate_train_test_split

def make_model(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        # Defining LMU layer
        lmu_layer = keras_lmu.LMU(
            memory_d=9,
            order=42,
            theta=1,
            hidden_cell=tf.keras.layers.SimpleRNNCell(units=500),
            )(conv3)
        #lmu_layer = keras_lmu.LMU(
        #    memory_d=9,
        #    order=42,
        #    theta=1,
        #    hidden_cell=tf.keras.layers.SimpleRNNCell(units=500),
        #    )(conv3)

        #lmus = lmu_layer(input_layer)
        #input_layer()
        output_layer = Dense(num_classes, activation="softmax")(lmu_layer)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)
callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model_lmu.h5", save_best_only=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
train_xs_array = []
test_xs_array  = []

train_ys_array = []
test_ys_array  = []

predict_ys_array = []

data_dir = '../fall_detection_data/merged_processed/'
subjects = []
for i in range(6,39):
    subjects.append('SA{:02d}'.format(i))

for subject in subjects:
    subject_file = '{}-merged.csv'.format(subject)

    # LDN parameters
    size_in = 9         # 9 features of the fall detection data
    theta = 1
    q = 42

    # simulation parameters
    dt = 0.01

    # load data and generate the train/test split
    ddf = pd.read_csv(os.path.join(data_dir,subject_file),index_col=0).drop(['TimeStamp(s)','FrameCounter'],axis=1)
    #print(ddf)

    train_test_split = 0.8
    chunk_size = 3600



    chunk_indices = np.arange(0,ddf.shape[0],chunk_size)
    train_chunk_indices = np.random.choice(chunk_indices,size = int(len(chunk_indices)*train_test_split), replace = False)
    test_chunk_indices = list(set(chunk_indices)-set(train_chunk_indices))

    train_df = pd.DataFrame(columns=ddf.columns)
    for idx in train_chunk_indices:
        train_df = pd.concat([train_df,ddf.iloc[idx:idx+chunk_size,:]],axis=0)

    test_df = pd.DataFrame(columns=ddf.columns)
    for idx in test_chunk_indices:
        test_df = pd.concat([test_df,ddf.iloc[idx:idx+chunk_size,:]],axis=0)

    train_xs = train_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].to_numpy().astype(float)
    lmu_train_xs = LDN(theta=theta, q=q, size_in=size_in).apply(train_xs)
    train_ys = train_df[['Fall/No Fall']].to_numpy().astype(int)

    test_xs = test_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].to_numpy().astype(float)
    test_ys = test_df[['Fall/No Fall']].to_numpy().astype(int)


    #root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    #
    #x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    #x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    num_classes = len(np.unique(train_ys))
    train_xs = train_xs.reshape((train_xs.shape[0], train_xs.shape[1], 1))
    test_xs = test_xs.reshape((test_xs.shape[0], test_xs.shape[1], 1))

    idx = np.random.permutation(len(train_xs))
    train_xs = train_xs[idx]
    train_ys = train_ys[idx]


    train_xs_array.append(train_xs)
    test_xs_array.append(test_xs)

    train_ys_array.append(train_ys)
    test_ys_array.append(test_ys)
    #print(x_train.shape[1:])
    #print(train_xs.shape[1:])
    model = make_model(input_shape=test_xs.shape[1:])
    keras.utils.plot_model(model, show_shapes=True)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
        )
    epochs = 1
    batch_size = 256



    history = model.fit(
        train_xs,
        train_ys,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    model = keras.models.load_model("best_model_lmu.h5")
    #test_loss, test_ac/subjectuuc = model.evaluate(test_xs, test_ys)
    predict_ys = model.predict(test_xs)
    predict_ys_array.append(predict_ys)

for i in range(len(test_xs)):
    predictions = np.argmax(predict_ys_array[i], axis = 1)
    predictions_score = np.argmax(predict_ys_array[i], axis =1) * np.max(predict_ys_array[i], axis = 1) / np.sum(predict_ys_array[i], axis=1)
    time = np.arange(test_xs_array[i].shape[0])
    fpr,tpr,thr = roc_curve(y_true = test_ys_array[i], y_score = predictions_score,pos_label=1)

    #tn, fp, fn, tp = confusion_matrix(test_ys, np.amax(predictions.flatten(), axis=0).ravel())

    #for performance_metric, number in zip(('True Negatives','False Positives','False Negatives','True Positives'),(tn, fp, fn, tp)):
        #print('{}: {}'.format(performance_metric,number))

    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    time = np.arange(len(test_xs_array[i]))
    ax1.plot(time, test_ys_array[i].flatten(), label='True')
    #ax2.plot(time, [p_category]/scaling_factor, label='Predicted')
    ax2.plot(time, np.argmax(predictions, axis = 1).flatten(), color="orange", label = 'Predicted')

    ax1.legend()
    ax2.legend()
    #plt.show()
    #plt.figure()
    #plt.plot(time, y_predict)
    #plt.plot(time, test_ys)
    plt.show()
    #print("Test accuracy", test_acc)
    #print("Test loss", test_loss)
    #
    #metric = "sparse_categorical_accuracy"
    #plt.figure()
    #plt.plot(history.history[metric])
    #plt.plot(history.history["val_" + metric])
    #plt.title("model " + metric)
    #plt.ylabel(metric, fontsize="large")
    #plt.xlabel("epoch", fontsize="large")
    #plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
