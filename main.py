from deepface import DeepFace
import json


def face_verification():

    result = DeepFace.verify(img1_path='Jolie/jol2.jpg',
                             img2_path='Jolie/jol3.jpg')

    return result


def face_recognition():

    result = DeepFace.find(img_path='Jolie/jol1.jpg',
                           db_path='Angelina')

    result = result.values.tolist()

    return result


def face_analysis():

    result = DeepFace.analyze(img_path='Angelina/ang1.jpg',
                              actions=['age', 'gender', 'race', 'emotion'])

    return result


with open('result.json', 'w') as file:
    json.dump([face_verification(), face_recognition(), face_analysis()],
              file, indent=4, ensure_ascii=False)
