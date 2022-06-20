## Facial Recognition

<p align="center"><img src="DeepFace/Facial Recognition.png" width="500" height="340"></p>

A very cool module that allows you to find and compare faces from images with high accuracy and in just a few lines of code. Among other things, it can analyze emotions, determine age, distinguish between men and women, as well as recognize race.

## Installation

Let's install the deepface library, it has some heavy dependencies, so we'll have to wait a bit.

```shell
$ pip install deepface
```

Then you will be able to import the library and use its functionalities.

```python
from deepface import DeepFace
```

 While Deepface handles all these common stages in the background, you don’t need to acquire in-depth knowledge about all the processes behind it. You can just call its verification, find or analysis function with a single line of code.

**Face Verification**

This function verifies face pairs as same person or different persons. It expects exact image paths as inputs. Passing numpy or based64 encoded images is also welcome. Then, it is going to return a dictionary and you should check just its verified key.

```python
result = DeepFace.verify(img1_path='Jolie/jol2.jpg', img2_path='Jolie/jol3.jpg')
```

<p align="center"><img src="DeepFace/Face Verification.png" width="50%" height="50%"></p>

**Face recognition**

Face recognition requires applying face verification many times. Herein, deepface has an out-of-the-box find function to handle this action. It's going to look for the identity of input image in the database path and it will return pandas data frame as output.

```python
result = DeepFace.find(img_path='Jolie/jol1.jpg', db_path='Angelina')
```
DataFrame is not very convenient to work with, so we can very easily turn it into a regular list. We turn to the values property, and then to the tolist() function.

```python
result = result.values.tolist()
```

<p align="center"><img src="DeepFace/Face recognition.png" width="60%" height="60%"></p>

**Facial Attribute Analysis**

Deepface also comes with a strong facial attribute analysis module including age, gender, facial expression (including angry, fear, neutral, sad, disgust, happy and surprise) and race (including asian, white, middle eastern, indian, latino and black) predictions.

```python
result = DeepFace.analyze(img_path='Angelina/ang1.jpg', actions=['age', 'gender', 'race', 'emotion'])
```

<p align="center"><img src="DeepFace/Facial Attribute Analysis.png" width="85%" height="85%"></p>

**Result**

To make it easier to read the data, I used the json module.

```python
with open('result.json', 'w') as file:
    json.dump([face_verification(), face_recognition(), face_analysis()], file, indent=4, ensure_ascii=False)
```

**P.S.**

Please note the warning:

```shell
Representations stored in  Angelina / representations_vgg_face.pkl  file. Please delete this file when you add new identities in your database.
```
It implies that if we add a new image to our catalog with a person's face, we have to restart the function to build a new file with views, previously deleting the old one, which was created during the first run and is in the catalog with the dataset.
