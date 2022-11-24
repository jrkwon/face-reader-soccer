# face-reader-soccer

![image](test_images/02.jpg)

This is an implementation of https://colab.research.google.com/drive/1H-AJX_URzN1fJn1Vhjh1_ehwc5Ih7eTe?usp=sharing (Google CoLab) on a local machine. If you have a machine with GPU, let's run the code on your machine.

## Environment Setup

```bash
$ conda env create -f environment.yaml
```

## Folder Structure

```
- dataset/imgs/*.png
- model/face-reader-soccer.h5 # trained model for this face reader
- model/model.tflite # pretrained model for ArcFace
- test_images/nn.jpg # images for testing face reader
```

## Data Folder

Files can be downloaded from https://drive.google.com/drive/folders/1nVU7dO_3X_mwLoQ11hVR6jZ7zPmaoNIi?usp=share_link

- `dataset/imgs.zip` must be unzip before using it.
- `model/model.tflite`: When an object of ArcFace class is instantiated, this model is supposed to be downloaded. Yet, I see time-out errors before the download ends. I manually downloaded this pretrained model file and the model path must be specified when ArcFace class is instantiated.
```python
face_rec = ArcFace.ArcFace(model_path='model/model.tflite')
```

## Test
- Activate `face-reader-soccer` conda environment.
```bash
$ conda activate face-reader-soccer
```
- Run `face-reader-soccer.py`
```bash
(face-reader-soccer) $ python face-reader-soccer.py
```
- You will see who is the most likely best soccer player among guys in the `test_images` directory.

![image](test_images/crop00.jpg)
![image](test_images/crop01.jpg)
![image](test_images/crop02.jpg)
![image](test_images/crop03.jpg)
![image](test_images/crop04.jpg)