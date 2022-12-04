import os
from pathlib import *
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")
import datetime

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, applications, backend, utils
from tensorflow.keras.preprocessing import image


from sklearn import preprocessing
from sklearn import metrics

from django.shortcuts import render, redirect
from django.views.generic import CreateView

from avanPostProj import settings
from .forms import New_DataSet, GetPhotoTest


def retrain(request):
    if request.method == 'POST':
        number_a = len(os.listdir(Path(Path.cwd(), 'neural', 'hack', 'hack')))
        os.mkdir(Path(Path.cwd(), 'neural', 'hack', 'hack', f'new_hack{number_a}'))

        form = New_DataSet(request.POST, request.FILES)
        if form.is_valid():
            return render(request, 'neural/Success.html')

            # a = request.FILES.getlist('all_files')
            # num = 0
            # for i in a:
            #     path = f"{Path(Path.cwd(), 'neural', 'hack', 'hack', f'new_hack{number_a}', str(num))}.jpg"
            #     destination = open(path, 'wb+')
            #     for chunk in i.chunks():
            #         destination.write(chunk)
            #         num += 1
            #     destination.close()

            # if retrain_model():
            #     return render(request, 'neural/result.html', {"res": ""})

        else:
            print(form.errors.as_data())

    else:
        form = New_DataSet()
    return render(request, 'neural/retrain.html', {'form': form})

def result(request, data):
    return render(request, "neural/result.html", data)

def index(request):
    if request.method == 'POST':
        for filename in os.listdir(Path(Path.cwd(), 'neural', 'new_neural', 'new_neural')):
            os.remove(Path(Path.cwd(), 'neural', 'new_neural', 'new_neural', filename))
        form = GetPhotoTest(request.POST, request.FILES)
        if form.is_valid():
            destination = open(Path(Path.cwd(), 'neural', 'new_neural', 'new_neural', 'file.jpg'), 'wb+')
            for chunk in request.FILES['file'].chunks():
                destination.write(chunk)
            destination.close()
            res = make_prediction()
            return render(request, 'neural/result.html', {"res": res})

        else:
            print(form.errors.as_data())
    else:
        form = GetPhotoTest()
        return render(request, 'neural/index.html', {'form': form})


def make_prediction():
    IMAGE_SHAPE = (226, 226)
    BATCH_SIZE = 500
    IMAGE_PATH = Path(Path.cwd(), 'neural', 'new_neural')
    train_datagen = image.ImageDataGenerator(
        rotation_range=30,  # угол поворота
        zoom_range=0.2,  # изменение масштаба
        horizontal_flip=True,  # зеркальное отображение
        validation_split=0.20
    )
    train_generator = train_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode='categorical',
        subset='training'
    )
    targets = [i for i in os.listdir(Path(Path.cwd(), 'neural', 'hack', 'hack'))]
    # targets = list(train_generator.class_indices.keys())
    paths = Path(Path.cwd(), 'model_t', 'best_model')
    model = tf.keras.models.load_model(paths)
    pred = model.predict_generator(train_generator)
    sc = np.argmax(pred, axis=1)
    return targets[sc[0]]


def retrain_model():
    IMAGE_SHAPE = (226, 226)
    BATCH_SIZE = 500
    COUNT_OUTPUT = len(os.listdir(Path(Path.cwd(), 'neural', 'hack', 'hack')))
    IMAGE_PATH = Path(Path.cwd(), 'neural', 'hack', 'hack')
    MODEL_PATH = Path(Path.cwd(), 'model_t', 'best_model')
    train_datagen = image.ImageDataGenerator(
        rotation_range=30,  # угол поворота
        zoom_range=0.2,  # изменение масштаба
        horizontal_flip=True,  # зеркальное отображение
    )
    train_generator = train_datagen.flow_from_directory(
        IMAGE_PATH,
        target_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode='categorical',
        subset='training'
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    model = tf.keras.models.Sequential(model.layers[:-1])
    model.add(layers.Dense(COUNT_OUTPUT, activation='softmax'))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_generator,
        epochs=10
    )
    model.save(MODEL_PATH)
    return 1