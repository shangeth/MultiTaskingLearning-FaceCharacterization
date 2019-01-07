import glob
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import os.path


def get_txt_files():
    txt_files = glob.glob('**/**/**/*.txt', recursive=True)
    txt_files_path = ['/home/shangeth/Desktop/Projects-DL/FaceDetection/' + s for s in txt_files]
    return txt_files_path

def txt_to_csv(files):
    common_df = pd.read_csv(files[0], sep='\t', header=None)
    l=[]
    for i, file in enumerate(files[1:]):
        print('{}/{}'.format(i, len(files)))
        try:
            print(file)
            df = pd.read_csv(file, sep='\t', header=None)
            common_df = common_df.append(df)

        except:
            print('ERROR in ',file)
            l.append(file)
            continue
        common_df.to_csv('face_data.csv')
        pd.DataFrame(l).to_csv('ERROR.csv')

def clean_csv():
    df = pd.read_csv('face_data.csv')
    print(df.head())
    cols = [0, 1, 8, 11, 12, 16, 17]
    df.drop(df.columns[cols], axis=1, inplace=True)
    print(df.head())
    df.to_csv('final_faces.csv')

def visualize_sample():
    df = pd.read_csv('face_data.csv')
    test = df.iloc[2870]
    path = '/home/shangeth/Desktop/Projects-DL/FaceDetection/IMFDB_final/'+test[
        'actor']+'/'+test['movie']+'/images/'+test['name']

    sex = test['sex']
    emotion = test['emotion']
    age = test['age']
    face_direction = test['face_direction']

    print(path)
    print(set(df['emotion'].values))


    img = Image.open(path)
    d = ImageDraw.Draw(img)
    d.text((0, 0), sex+'\n'+emotion+'\n'+age+'\n'+face_direction, fill=(255, 255, 0))


    plt.imshow(img)
    plt.show()

def final_dataset():
    df = pd.read_csv('face_data.csv')
    row=[]
    for i in range(len(df)):
        print('{}/{}'.format(i,len(df)))
        try:
            test = df.iloc[i]
            path = '/home/shangeth/Desktop/Projects-DL/FaceDetection/IMFDB_final/' + test['actor'] + '/' + test['movie'] + '/images/' + test['name']
            sex = test['sex']
            emotion = test['emotion']
            age = test['age']
            face_direction = test['face_direction']
            if os.path.isfile(path):
                row.append([path, sex, emotion, age, face_direction])
            else:
                continue
        except:
            print('ERROR in {}'.format(i))
            continue
    new_df = pd.DataFrame(row, columns=['image', 'sex', 'emotion', 'age', 'face_direction'])
    new_df.to_csv('final_csv_data.csv')

def find_image_size():
    lst=[]
    df = pd.read_csv('final_csv_data.csv')
    for i in range(len(df)):
        path = df.iloc[i]['image']
        img = Image.open(path)
        h, w = img.size
        lst.append(min(h,w))
    return(lst)

def split_data_on_size():
    df = pd.read_csv('final_csv_data.csv')
    list50=[]
    list100=[]
    list128=[]
    list200=[]
    list224=[]
    for i in range(len(df)):
        path = df.iloc[i]['image']
        h, w = Image.open(path).size
        l = min(h, w)
        if l >= 50:
            list50.append(df.iloc[i])
        if l >=100:
            list100.append(df.iloc[i])
        if l >= 128:
            list128.append(df.iloc[i])
        if l >= 200:
            list200.append(df.iloc[i])
        if l >= 224:
            list224.append(df.iloc[i])
    pd.DataFrame(list50).reset_index(drop=True).to_csv('list50.csv', index=None)
    pd.DataFrame(list100).reset_index(drop=True).to_csv('list100.csv', index=None)
    pd.DataFrame(list128).reset_index(drop=True).to_csv('list128.csv', index=None)
    pd.DataFrame(list200).reset_index(drop=True).to_csv('list200.csv', index=None)
    pd.DataFrame(list224).reset_index(drop=True).to_csv('list224.csv', index=None)


if __name__ == '__main__':
    # files = get_txt_files()
    # data_csv = txt_to_csv(list(set(files)))
    # clean_csv()
    # visualize_sample()
    # final_dataset()
    # lst = find_image_size()
    split_data_on_size()
