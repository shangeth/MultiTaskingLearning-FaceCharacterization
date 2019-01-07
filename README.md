# Face Detection and Characterization with Deep Learning
Using deep learning regression technique to fing the faces in a image and using multi tasking 
learning to find some other characteristics of faces like race, age, position.


## Data set
IMFDB images dataset is used for this project, which contains all the indian actors faces and 
arrtibutes of it.

## Data Preparation
### gathering the files
getting all the txt files containing the characteristics of each image in all the folders.
```get_txt_files()``` function is used to get the list og all txt files in all the sub 
directories using ```glob``` package.
```python
def get_txt_files():
    txt_files = glob.glob('**/**/**/*.txt', recursive=True)
    txt_files_path = ["Main Folder" + s for s in txt_files]
    return txt_files_path

```

### text to csv and handling error in data
We need to convert all the txt files to csv files for easy accessibility using dataframes.
And we need to make sure all the image files in the csv files exists and all the attribute exists
 for each image.THis is done by 
 ```python
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
```
<strong>result:</strong> 

```ERROR.csv``` file contains all the image with error in data.
```face_data.csv``` contains all the valid data.

### getting the necessary data
Removing un needed columns in the dataframe and getting the actual path of the image in the local
 system and their attributes like sex, age, emotion, face direction.
 ```python
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
    path = 'main folder'+test[
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
            path = 'main folder' + test['actor'] + '/' + test['movie'] + '/images/' + test['name']
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
```

<strong> result:</strong>  ```final_csv_data.csv```  contains the location of the image int he 
disk and the needed attributes.

### seperating data by Image size 
I am not sure of what model architecture to use for this data , as the image size was distributed
 equally from size of 50 to 300.
 
We don't want to use image size of 50 and resize it to 224 , if we decide to use imagenet 
pretrained model for transfer learning.

So we seperate data into image size >50, >100, >128, >200 & >=224 . and we try out imagent models
 and our own model for image size <224.
 
 ```python
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
```

## Image data processing
As we are using different image sizes of >50, >100, >128, >200, >=224, we want to resize the 
images into a single size for each category.

The image has been :
1. Normalized ```norm_image()```
2. Resized and Padded ```resize_image()```


## MultiTasking Model

![](readme_assests/mtl_images-001-2.png)

We use transfer learning to learn the features fo the image and use those features for each of 
our specific tasks like determining age, emotion, sex, face_direction and face deteion will be 
added later.

The base network is a pretrained VGG19 architecture , and we add 4 other networks for out 
classification tasks.

The Computation graph will look like above. and the final loss function will be average of losses 
of individual tasks.

I have used cross entropy as the loss function for each of the task.

### Model in Pytorch
```python
class MultiTaskLearning(nn.Module):
  def __init__(self, convbase):
    super(MultiTaskLearning, self).__init__()
    self.base = convbase
    
    classifier_input_size = 1000
    classifier1_output_size = 2
    classifier2_output_size = 7
    classifier3_output_size = 3
    classifier4_output_size = 5
    
    self.classifier1 = nn.Sequential(OrderedDict([
    ('c1_fc1', nn.Linear(classifier_input_size, 512)),
    ('c1_relu', nn.ReLU()),
    ('c1_fc2', nn.Linear(512, 128)),
    ('c1_relu', nn.ReLU()),
    ('c1_fc3', nn.Linear(128, classifier1_output_size)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
    self.classifier2 = nn.Sequential(OrderedDict([
    ('c2_fc1', nn.Linear(classifier_input_size, 512)),
    ('c2_relu', nn.ReLU()),
    ('c2_fc2', nn.Linear(512, 128)),
    ('c2_relu', nn.ReLU()),
    ('c2_fc3', nn.Linear(128, classifier2_output_size)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
      
    self.classifier3 = nn.Sequential(OrderedDict([
    ('c3_fc1', nn.Linear(classifier_input_size, 512)),
    ('c3_relu', nn.ReLU()),
    ('c3_fc2', nn.Linear(512, 128)),
    ('c3_relu', nn.ReLU()),
    ('c3_fc3', nn.Linear(128, classifier3_output_size)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
        
    self.classifier4 = nn.Sequential(OrderedDict([
    ('c4_fc1', nn.Linear(classifier_input_size, 512)),
    ('c4_relu', nn.ReLU()),
    ('c4_fc2', nn.Linear(512, 128)),
    ('c4_relu', nn.ReLU()),
    ('c4_fc3', nn.Linear(128, classifier4_output_size)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
  def forward(self, x):
    x_base = self.base(x)
    clf1_output = self.classifier1(x_base)
    clf2_output = self.classifier2(x_base)
    clf3_output = self.classifier3(x_base)
    clf4_output = self.classifier4(x_base)
    return clf1_output, clf2_output, clf3_output, clf4_output     
     
```
```python
model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
  param.required_grad = True
    
modelnew = MultiTaskLearning(model)
```
### Training in Pytorch(without Augmentation
<pre>
Epoch - 1
############
Loss(Sex) : 0.7412909052588723
Loss(Emotion) : 2.242590586751432
Loss(Age) : 1.1959744549896592
Loss(Face Direction) : 2.1285165858177155
Loss(Total) : 1.5770931195052729
________________________________________

Train acc(Sex) : 0.5423910021781921
Train acc(Emotion) : 0.024575572460889816
Train acc(Age) : 0.1598750650882721
Train acc(Face Direction) : 0.5855371356010437
Train acc(Total) : 0.3280947208404541
________________________________________

Epoch - 2
############
Loss(Sex) : 0.7394668456076354
Loss(Emotion) : 2.2383922523176167
Loss(Age) : 1.193567744595751
Loss(Face Direction) : 2.1313986073077564
Loss(Total) : 1.5757063637927613
________________________________________

Train acc(Sex) : 0.5421967506408691
Train acc(Emotion) : 0.025249920785427094
Train acc(Age) : 0.15991245210170746
Train acc(Face Direction) : 0.5879499912261963
Train acc(Total) : 0.32882726192474365
________________________________________
.
.
.
</pre>
### Model in Keras
```python
inputs = Input(shape=(224, 224, 3), name='inputs')
conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))(inputs)
flatten = Flatten()(conv_base)

x_sex = Dense(1024, activation='relu')(flatten)
x_sex = Dropout(0.4)(x_sex)
x_sex = Dense(512, activation='relu')(x_sex)
x_sex = Dropout(0.5)(x_sex)
x_sex = Dense(256, activation='relu')(x_sex)
x_sex = Dropout(0.4)(x_sex)
x_sex = Dense(128, activation='relu')(x_sex)
x_sex_output = Dense(1, activation='sigmoid', name='x_sex_output')(x_sex)

x_emotion = Dense(1024, activation='relu')(flatten)
x_emotion = Dropout(0.4)(x_emotion)
x_emotion = Dense(512, activation='relu')(x_emotion)
x_emotion = Dropout(0.5)(x_emotion)
x_emotion = Dense(256, activation='relu')(x_emotion)
x_emotion = Dropout(0.4)(x_emotion)
x_emotion = Dense(128, activation='relu')(x_emotion)
x_emotion = Dropout(0.4)(x_emotion)
x_emotion = Dense(64, activation='relu')(x_emotion)
x_emotion_output = Dense(y_emotion_hot.shape[1], activation='softmax', name='x_emotion_output')(x_emotion)

x_age = Dense(1024, activation='relu')(flatten)
x_age = Dropout(0.4)(x_age)
x_age = Dense(512, activation='relu')(x_age)
x_age = Dropout(0.5)(x_age)
x_age = Dense(256, activation='relu')(x_age)
x_age = Dropout(0.4)(x_age)
x_age = Dense(128, activation='relu')(x_age)
x_age_output = Dense(y_age_hot.shape[1], activation='softmax', name='x_age_output')(x_age)

x_fd = Dense(1024, activation='relu')(flatten)
x_fd = Dropout(0.4)(x_fd)
x_fd = Dense(512, activation='relu')(x_fd)
x_fd = Dropout(0.5)(x_fd)
x_fd = Dense(256, activation='relu')(x_fd)
x_fd = Dropout(0.4)(x_fd)
x_fd = Dense(128, activation='relu')(x_fd)
x_fd_output = Dense(y_fd_hot.shape[1], activation='softmax', name='x_fd_output')(x_fd)
```
## Model Prediction
```python
idxdict = [{0:'Female', 1:'Male'}, 
           {0:'Anger', 1:'Disgust',2:'Fear', 3:'Happiness', 4:'Neutral', 5:'Sadness', 6:'Surprise' }, 
           {0:'Middle Age', 1:'Old Age', 2:'Old Age' }, 
           {0:'Down', 1:'Frontal', 2:'Left', 3:'Right', 4:'Up'}]

def getmaxindex(l):
  a=[]
  if l[0][0] >0.5: 
    l[0]=np.array([[0,1]])
  else: l[0] = np.array([0,1])
  for i in range(len(l)):
    a.append(l[i][0].argmax(axis=0))
  return a

def predict_image(img):
  pred_txt=[]
  plt.imshow(img)
  plt.grid(False)
  plt.show()
  pred = model.predict(im.reshape(1, 224, 224, 3))
  print('Model Prediction :\n',pred)
  idx = getmaxindex(pred)
  for i in range(len(idx)):
    pred_txt.append(idxdict[i][idx[i]])
  print('\n',pred_txt)
```
![](https://i.pinimg.com/236x/b0/af/9a/b0af9abb797aea9ae658902e0b272594--christoph-waltz-supporting-actor.jpg)
<pre>
Model Prediction :
 [array([[0.98098755]], dtype=float32), array([[5.4912241e-31, 1.6996787e-19, 0.0000000e+00, 1.0000000e+00,
        8.5350787e-14, 6.1946490e-20, 3.0885606e-29]], dtype=float32), array([[1.146524e-21, 4.130661e-31, 1.000000e+00]], dtype=float32), array([[5.6413937e-01, 5.1520718e-04, 2.8015572e-01, 3.5165754e-04,
        1.5483807e-01]], dtype=float32)]

 ['Male', 'Happiness', 'Old Age', 'Down']</pre>
 
 ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRafPgJepZ9chCl9ZGLi0EC5e3mVmc05wPHvxuiYg4kCiFvINeBvQ)
 <pre>
 Model Prediction :
 [array([[4.9941887e-06]], dtype=float32), array([[3.3553686e-09, 3.8809858e-06, 1.2822178e-11, 9.9932766e-01,
        6.5777352e-04, 1.0612529e-05, 1.2714095e-08]], dtype=float32), array([[8.1378478e-01, 3.1865619e-07, 1.8621497e-01]], dtype=float32), array([[7.1780937e-04, 9.0216461e-05, 9.9868721e-01, 2.3750174e-06,
        5.0242245e-04]], dtype=float32)]

 ['Female', 'Happiness', 'Middle Age', 'Left']
 </pre>
![](https://d.wattpad.com/story_parts/558325776/images/1529737a4b9211c6445421122304.jpg)
<pre>
Model Prediction :
 [array([[0.00011555]], dtype=float32), array([[1.3009728e-03, 1.7888163e-04, 2.8493056e-07, 9.9974649e-04,
        9.9407601e-01, 2.8159078e-03, 6.2812644e-04]], dtype=float32), array([[0.72783643, 0.01925719, 0.25290638]], dtype=float32), array([[3.6774931e-04, 7.3594981e-01, 2.6367772e-01, 2.8114794e-06,
        1.8422278e-06]], dtype=float32)]

 ['Female', 'Neutral', 'Middle Age', 'Frontal']
 </pre>
 ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-Z-SxZyTnBDoM9ZFdogti0w_4i9liLxxoBJrOe4oqYvjmqVm4gQ)
 <pre>
 Model Prediction :
 [array([[1.]], dtype=float32), array([[1.8694135e-12, 9.8778297e-10, 2.0186993e-18, 4.8219282e-02,
        9.5177954e-01, 1.1466984e-06, 6.2040190e-10]], dtype=float32), array([[9.9999845e-01, 9.3176354e-08, 1.4640041e-06]], dtype=float32), array([[2.5171862e-04, 1.4977384e-03, 9.9817860e-01, 5.6700296e-06,
        6.6222106e-05]], dtype=float32)]

 ['Male', 'Neutral', 'Middle Age', 'Left']</pre>
 ![](https://usercontent2.hubstatic.com/7653579.jpg)
<pre>
Model Prediction :
 [array([[0.99999964]], dtype=float32), array([[6.8111354e-12, 1.0207035e-07, 4.7114721e-15, 9.9997067e-01,
        2.8971483e-05, 2.1610518e-07, 5.4456648e-11]], dtype=float32), array([[2.1819297e-09, 8.3214922e-14, 1.0000000e+00]], dtype=float32), array([[1.0881673e-05, 9.9939263e-01, 4.0517339e-06, 5.9169094e-04,
        7.8347290e-07]], dtype=float32)]

 ['Male', 'Happiness', 'Old Age', 'Frontal']
</pre>
## Future improvements to be made

1. Make a Pytorch Module for MultiTasking Learning & Augmentation for multitasking learning.
2. adding segmentation model to segment the face from an image or video .
3. using autoencoders to blur out or segment the face from the image or video.
4. develope a web app to demonstrate this project.
5. Get augmentation dataset 
