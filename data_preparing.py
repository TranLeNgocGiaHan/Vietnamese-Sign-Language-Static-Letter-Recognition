#%%
from pathlib import Path
import cv2
import numpy as np
import math
import os
import keras
from cvzone.HandTrackingModule import HandDetector
from sklearn.model_selection import train_test_split
import tensorflow as tf
import mediapipe as mp
from keras.models import Model
import matplotlib.pyplot as plt
#%%
current_path = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# data_path = current_path / "data"
lm_data_path = current_path / "lm_data_for_v5"
alphabet = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'mu', 'moc', 'do']
#%%
def createFilePath():
    for letter in alphabet:
        word_path = Path(lm_data_path / letter)
        word_path.mkdir(parents=True, exist_ok=True)
#%%
def saveLandmarks():
    OFFSET = 70
    IMG_SIZE = 300

    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=1, maxHands=1)
    index = 1
    letter = alphabet[index]

    while cap.isOpened():
            answer = input("Start collecting for {} (y/n):".format(letter))
            if answer.lower() == 'y':
                count = 0
                while count < 2000:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    frame = cv2.rectangle(frame, (400, 100), (600, 400), (255,0,0), 2)
                    img = frame[100:400, 400:600]
                    hands, img = detector.findHands(img, flipType=False)
                    
                    if hands:
                        hand = hands[0]
                        lm = hand['lmList']
                        # x, y, w, h = hand['bbox']
                        landmarks = np.array(lm, float).flatten()/255.0
                        npy_path = lm_data_path/ letter /  str(count)
                        np.save(npy_path, landmarks)
                        print(count)
                        count+=1
                        if count == 2000:
                            index+=1
                            letter=alphabet[index]
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1)

                    if key == ord("q"):
                        break
    cap.release()
    cv2.destroyAllWindows()
#%%
def loadDataFromPath():
    print("Loading data from path...")
    landmarks_list, labels = [], []
    for inx, letter in enumerate(alphabet):
        for file_num in range(2000):
            file_path = lm_data_path / letter / "{}.npy".format(str(file_num))
            res = np.load(str(file_path))
            landmarks_list.append(res)
            labels.append(inx)

    X = np.array(landmarks_list)
    y = keras.utils.to_categorical(labels).astype(int)
    return X, y
#%%
def splitTrainValidateTest(X, y):
    print("Splitting data into train, validate and test sets...")
    X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=5000, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=5000, stratify=y_main)
    return X_train, X_val, X_test, y_train, y_val, y_test
#%%
def defineModel():
    print("Defining model...")
    model = keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(63),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(25, activation='softmax')
    ])
    return model
#%%
def compileModel(model):
    print("Compiling model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#%%
def fitModel(model, X_train, X_val, y_train, y_val):
    print("Fitting model...")
    callBack = tf.keras.callbacks.EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=5
    )
    history = model.fit(X_train, y_train, epochs =100, batch_size = 64,validation_data=(X_val,y_val), callbacks=[callBack])
    return history
#%%
def evaluateModel(model, X_val, X_test, y_val, y_test):
    score = model.evaluate(x = X_test, y = y_test, verbose = 0)
    print('Accuracy for test images:', round(score[1]*100, 3), '%')
    score = model.evaluate(x = X_val, y = y_val, verbose = 0)
    print('Accuracy for validate images:', round(score[1]*100, 3), '%')
#%%
def plotModelAccuracyAndLoss(history):
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
#%%
def main():
    createFilePathYes = input("""
    Create file path?
    -> Press Y to create new
    -> Press N if file path already exists 
    """)
    
    if createFilePathYes.lower() == "y":
        createFilePath()
    
    saveLandmarksYes = input("""
    Start collecting landmarks?
    -> Press Y to start
    -> Press N if landmark files already exists 
    """)

    if saveLandmarksYes.lower() == "y":
        saveLandmarks()

    X, y = loadDataFromPath()
    X_train, X_val, X_test, y_train, y_val, y_test = splitTrainValidateTest(X, y)
    
    model = defineModel()
    compileModel(model)
    history = fitModel(model, X_train, X_val, y_train, y_val)

    evaluateModel(model, X_val, X_test, y_val, y_test)
    plotModelAccuracyAndLoss(history)
#%%
if __name__ == '__main__':
    main()
# %%
