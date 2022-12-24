# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
import requests
import shutil
import dlib
import pickle
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from arcface import ArcFace
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split


# %%
data_path = 'dataset'
data_imgs_path = f'{data_path}/imgs'
model_path = 'model'
pretrained_arcface_model = f'{model_path}/model.tflite'
model_name = f'{model_path}/face-reader-soccer.h5'
x_data_filename = 'x_data_file.pickle'
y_data_filename = 'y_data_file.pickle'

retrain = False # if you want to retrain the model, change this to True
model_loaded = False

if not os.path.exists(data_path):
    os.makedirs(data_path)
if not os.path.exists(data_imgs_path):
    os.makedirs(data_imgs_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


# %%
face_rec = ArcFace.ArcFace(model_path=pretrained_arcface_model)

# %%
detector = dlib.get_frontal_face_detector()


# %%
if retrain is False and os.path.exists(model_name) is True:
    model = keras.models.load_model(model_name)
    model.summary()
    model_loaded = True


# %%
if retrain is True and model_loaded is False:
    model = keras.Sequential([
        layers.Dense(256, input_shape=(512,), activation="relu"),
        layers.Dense(1),
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.summary()

# %% [markdown]
# ## Training

if os.path.exists(x_data_filename) is True:
    x_data_file = open(x_data_filename, 'rb')
    x_data = pickle.load(x_data_file)
    x_data_file.close()

    y_data_file = open(y_data_filename, 'rb')
    y_data = pickle.load(y_data_file)
    y_data_file.close()
else:

    img_list = glob(f'{data_imgs_path}/*.png')

    x_data = np.zeros((len(img_list), 512), dtype=np.float32)
    y_data = np.zeros((len(img_list), 1), dtype=np.float32)

    det_failed = 0
    for i, img_path in tqdm(enumerate(img_list)):
        img = Image.open(img_path).convert('RGB')

        dets = detector(np.array(img))

        if len(dets) == 0:
            # let's count how many detection failed.
            #print('failed in detection.')
            det_failed += 1
            continue

        det = dets[0]
        x1 = det.left()
        y1 = det.top()
        x2 = det.right()
        y2 = det.bottom()

        crop_img = img.crop((x1, y1, x2, y2))
        crop_img.save('temp_crop.jpg')

        emb = face_rec.calc_emb('temp_crop.jpg')

        overall = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[1])

        x_data[i] = emb
        y_data[i] = overall

    print("Face detection failed", det_failed, "times.")
    print(x_data.shape)
    print(y_data.shape)

    x_data_file = open(x_data_filename, 'wb')
    pickle.dump(x_data, x_data_file)
    x_data_file.close()

    y_data_file = open(y_data_filename, 'wb')
    pickle.dump(y_data, y_data_file)
    y_data_file.close()

# %%
if retrain is True and model_loaded is False:

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2022)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    history = model.fit(
        x_train,
        y_train,
        batch_size=256,
        epochs=200,
        validation_data=(x_val, y_val)
    )

    model.save(model_name)

    plt.figure()
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
else:
    print('A trained model was already loaded.')

# %% [markdown]
# ## Test with an image

# %%
test_img_path = 'test_images'
test_img_list = glob(f'{test_img_path}/??.jpg')


#plt.figure()
for image in test_img_list:
    img = Image.open(image)
#    plt.imshow(img)
#plt.show()


# %%
crop_images = []
for i, image in enumerate(test_img_list):
    img = Image.open(image)
    dets = detector(np.array(img))
    if len(dets) == 1:
        det = dets[0]

        x1 = det.left()
        y1 = det.top()
        x2 = det.right()
        y2 = det.bottom()

        crop_img = img.crop((x1, y1, x2, y2))
        crop_image_filename = f'{test_img_path}/crop{i:02d}.jpg'
        crop_images.append(crop_image_filename)
        crop_img.save(crop_image_filename)
        #plt.figure()
        #plt.imshow(crop_img)
        #plt.show()
    else:
        print('No face or multiple faces found.')

# %%
crop_images

# %%
how_likely = []
for i, crop in enumerate(crop_images):

    emb = face_rec.calc_emb(crop)
    emb = np.expand_dims(emb, axis=0)
    emb.shape

    how_likely.append(model.predict(emb)[0][0])

    print('Overall performance of a person having face {:} as a soccer player is predicted as {:.2f}.'.format(i, how_likely[-1]))

best_idx = how_likely.index(max(how_likely))
print('\nThe best soccer player based on face features is face {:} with {:.2f}'.format(best_idx, how_likely[best_idx]))
img = Image.open(crop_images[best_idx])
plt.figure()
plt.imshow(img)
plt.show()


