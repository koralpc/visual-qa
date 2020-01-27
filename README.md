# visual-qa
Multimodal LSTM + CNN Architecture implementation for Visual QA task

Hello there! Today, we will be building an extensive multimodal model, which will harmonize two well known Machine Learning models into one. We will be using the model we have built in a visual QA setup. More info is in the introduction!

## Table Of Contents

* [Introduction](#intro)
* [Parsing the Data](#parsing)
* [Constructing the Multimodal LSTM + CNN Model](#multimodal)
* [Accuracy Measures & Training the Network](#wups)
* [Results](#results)
* [Conclusion](#conclusion)


## Introduction <a name ="intro">

Multimodal models are an extension to the simple models that we work with. In these models, we have multiple channels of input and a single output. So, the aim of the model is to incorporate different level of knowledge into a single model, and do predictions accordingly.

However, as one can expect, more levels of medium means even harder tasks, so the accuracy for predictions in this project we will achieve will not be of an incredible extent. Although, the predictions will not be accurate, we will hope for them to be **meaningful**.

In this task, we will deal with visual question answering. VQA is an emerging concept, where we aim to teach the neural network to learn the encoded information in an input image, and a input text, so that it will be able to predict further questions about the same image. An example instance of the dataset looks like below:

![Example data](../assets/images/Post-assets/vqa-images/vqa-data.jpg)

Here, as we see, the input is a pair of image and a related question. For this specific image, the true label is **bed**. So, we want our network to learn that.


## Parsing the Data <a name ="parsing">

This step in our project is of high importance, because correctly processing the data into a training-ready format will enable us to actually perform the training of the model. However, this part is also divided into two steps:

* Parsing of text data
* Extracting image features


### Parsing Text Data

To begin with, our training dataset is of the form:

```
what is on the right side of the black telephone and on the left side of the red chair in the image3 ?
desk
what is in front of the white door on the left side of the desk in the image3 ?
telephone
what is on the desk in the image3 ?
book scissor papers tape_dispenser
what is the largest brown objects in this image3 ?
carton
what color is the chair in front of the white wall in the image3 ?
red
```

So, every 2 lines is a question-answer pair, which denotes an instance in our overall dataset. Hence, we start by grouping these pairs into individual arrays. For that, we use the slicing functionality of Python:

```python
#Read the file
def readFile(filepath):
    #Read all the lines in the file
    lines = [line.rstrip('\n') for line in open(filepath)]

    #Seperate questions and answers into different lists
    questions = lines[::2]
    answers   = lines[1::2]
    return [questions,answers]
```

Now, the next step is to extract the image information from questions. In each question, the image that the information is asked from is referred. For example, in  `what is on the desk in the image3 ?` , we know that the associated image with this instance will be the third index in our image features.

Therefore, in order to use this information, we store the image index that is used in each question in a different array for further referencing. For this, we do a String splitting operation via this function:

```python
#Seperate image tag from questions
def seperateImages(data):
    questions = []
    images = []
    j = 0
    for i in data:
        sp = i.split(' ')
        questions.append(' '.join(sp[:-4] + sp[:-2:-1]))
        images.append(sp[-2])
        j += 1
    return [questions,images]
```

#### Creating vocabulary

Using the data instances we have for the training, we can construct a vocabulary out of it. First, we count the frequency of each word in the text, using an external library called `toolz`. Then, using these frequency values, we construct our dictionary/vocabulary. The code for the vocabulary creation is below. The part for counting frequencies will be added in the main function. Note that the first 4 keys are for reserved keywords. Which are:

* 0: Pad value (Masked by network)
* 1: Unknown word
* 2: End of answer
* 3: End of question

```python
#Create vocabulary from the word counts
def createVocabulary(freqs):
    #The first 4 entries for reserved keywords
    vocabulary = [('<pad>',0),('<unk>', 1),('<eoa>', 2),('<eoq>', 3)]
    #Sort the list by occurence
    sorted_freqs = sorted(freqs.items(), key=operator.itemgetter(1) , reverse = True)
    j = 4
    for i,v in enumerate(sorted_freqs):
        vocabulary.append((sorted_freqs[i][0],j))
        j += 1
    return dict(vocabulary)
```

#### Encoding & Padding

The final step for text input, as given in the header, is to encode it into a numerical vector and then pad it with 0's up to a fixed length. The fixed length is important, because if we had variable size input, the training process of our network would be not realizable.

In the encoding part, we have two functions. One that encodes questions, and the other one that encodes answers. This is because, questions are encoded as sparse vectors, whereas answers are encoded as one-hot vectors. Moreover, the code for both encoders are below.

```python
#Encode questions with the vocabulary(Word 2 Integer encoding)
def encodeQuestions(data,vocabulary):
    encoded = []
    for i in data:
        temp = []
        x = i.split(' ')
        #For all words in the question, do the encoding
        for j in x:
            if j in vocabulary:
                temp.append(vocabulary[j])
            else:
                #Put unknown token if word not in vocabulary
                temp.append(vocabulary['<unk>'])
        temp.append(vocabulary['<eoq>'])
        encoded.append(temp)
    return encoded

#Encode answers with the vocabulary(One-hot encoding)
def encodeAnswers(data,vocabulary):
    encoded = np.zeros((len(data),len(vocabulary.keys())))
    j = 0
    for i in data:
        temp = []
        x = i.split(' ')
        #Only take first answer
        if x[0] in vocabulary:
            encoded[j][vocabulary[x[0]]] = 1
        else:
            #Put unknown token if word not in vocabulary
            encoded[j][vocabulary['<unk>']] = 1
        j += 1
    return encoded
```

Sequence padding after this point is relatively easy, because we can use a `Keras` library for that task.

```python
#Sequence padding to have a uniform length over all inputs
def sequencePad(data,MAXLEN = 30):
    padded = sequence.pad_sequences(data, maxlen=MAXLEN)
    return padded
```

Effectively, at this point we have the helper functions for parsing of text and we can move to parsing of image features.

### Extracting Image Data

For this task, the original dataset is readily found. However, using the raw images to incorporate the image information into training would both slow down our training process and also reduce the overall accuracy as we would need to build a very complex CNN to interpret the images alone well.

Therefore, rather than using raw images, we will be using extracted features of a CNN. So, we will use an intermediate layer output from a state-of-art CNN architecture that was trained over ImageNet dataset. The exact specifics at this point are too granular for the course of this project, so will be skipped.

Moreover, the dataset of extracted features can be found on my github.

#### Parsing the JSON File

Now, the JSON file will be available in my GitHub repo, so you can download it from there directly.

The JSON file stores the values that are extracted from an intermediate layer of ResNet. We will use these features directly as our training images, since they work as an embedding solution for us, and this can be useful to increase accuracy.

Hence, we write three helper functions to extract the values from a JSON file and structure them in an array.

```python
#Read the file
def readFile(filepath):
    #Read all the lines in the file
    lines = [line.rstrip('\n') for line in open(filepath)]

    #Seperate questions and answers into different lists
    questions = lines[::2]
    answers   = lines[1::2]
    return [questions,answers]

#Read the JSON data for features
def readJsonData(filepath):
    input_file=open(filepath, 'r')
    json_decode=json.load(input_file)
    return json_decode

#Convert the JSON dictionary to numpy array
def createDatasetFromJson(json_file):
    ret_list = []
    for key in json_file.keys():
        temp = json_file[key]
        ret_list.append(temp)
    return np.array(ret_list)
```

Although now we have the array in hand, there is still an option that is remaining. In our image array, each image occurs only once, whereas in the Question-Answer dataset, a query to an image can occur multiple times. Therefore, we will need to address to an image multiple times during training.

Since indexing and retrieving images during training time would be inefficient, we are gonna modify our training dataset, such that at each index of the question, there will be a corresponding image. To summarize that, we will duplicate our images w.r.t image queried in the respective question.


Therefore, we need another helper function to achieve that

```python
#Duplicate the used images
def duplicateImages(label_array,img_array):
    retArr = []
    for i in range(0,len(label_array)):
        #Minus one because the first index of an array is 0
        index = int(label_array.iat[i,0].replace('image','')) - 1
        retArr.append(img_array[index])
    #retArr2 = np.array(retArr)
    return np.array(retArr)
```


### Running the Parser

We have implemented the necessary operators for both sections of our dataset. Now, the final step is to put them together in a run, which can be called from outside and would be agnostic to the dataset.

The run() function will just be the logical sequence of the operators put in order. Hence, the implementation of the function will be as below.

```python
#Run all operations together - Used for exporting the class
def run(filename,vocab = 'default',vocab_ans = 'default'):

    #Read the data
    questions,answers = readFile(filename)

    #Create DataFrame from the lists
    df1,df3 = seperateImages(questions)

    df1 = pd.DataFrame(df1,columns = [ 'question'])
    df2 = pd.DataFrame(answers , columns = [ 'answer'])
    df3 = pd.DataFrame(df3,columns = [ 'images'])

    #Concatanate dataframes along their rows, and construct the final version of the dataframe
    frames = [df1,df2,df3]
    data = pd.concat(frames, axis = 1)

    #Check the first few elements for correctness
    data.head()

    #Count frequencies
    freqs_que = frequencies(' '.join(data['question']).split(' '))
    freqs_ans = frequencies(' '.join(data['answer']).split(' '))

    #Initialize the vocabulary - For test data, the training vocabulary is used. That's why the 'default' keyword is used.
    if vocab == 'default':
        vocabulary_que = createVocabulary(freqs_que)
    else:
        vocabulary_que = vocab

    if vocab_ans == 'default':
        vocabulary_ans = createVocabulary(freqs_ans)
    else:
        vocabulary_ans = vocab_ans

    #Encode question into integer vectors
    encoded_questions = encodeQuestions(data['question'],vocabulary_que)
    #Encode answers into one-hot vectors
    encoded_answers = encodeAnswers(data['answer'], vocabulary_ans)

    #Pad the questions into uniform length
    padded_questions = sequencePad(encoded_questions)
    #padded_answers = sequencePad(encoded_answers,MAXLEN = 2)

    return [df3,padded_questions,encoded_answers,vocabulary_que,vocabulary_ans]
```

This function returns couple of variables:

* Label of each image used in the corresponding questions
* Numerated + Zero padded question vectors
* One-hot encoded answers
* Question vocabulary
* Answer vocabulary  

---
All these terms will be in use either in our training process, or in our prediction process. Therefore, it is important to retrieve those after the operations. From this point, the next step will be to implement the multimodal architecture


## Constructing the Multimodal LSTM + CNN Model <a name="multimodal">

In our project, we have two channels of input data: text and image. Therefore, we will also have two neural network architectures that corresponding to the individual channels. These are :

* LSTM : Will be used to learn the semantics of the questions and answers
* CNN : Will be used to learn the encoded features in the images

However, one important fact is that, the models will not be trained individually, but rather will be trained concurrently. Therefore, their outputs will determine the end prediction together, hence this process is not equal to training individual models and combining their output prediction!

For more information about this, I would suggest to have a look at the theory behind multimodal models and their work principles.


Moreover, we have 2 input channels stated above, but only 1 output label, which are the answers. Therefore, after the predictions, we would need to combine the individual predictions and propose them as the overall prediction.


### Custom Neural Network Model

Since Keras does not have a direct layer function/model that corresponds to our needs, we will be writing our custom class. So, we will have the following class implementation

```python
class MergedModel:

    #Boolean pre-trained
    isPreTrained = False

    #Configuration is done via Config_dict dictionary
    Config_dict = []
    rnnModel = Sequential()
    visualModel = Sequential()
    genericModel = Sequential()
    funcModel = Model()

    #Datasets
    x = []
    y = []
    tx = []
    ty = []

    #Constructor
    def __init__(self,Config):
        print('\nHello, this is the Merged Model( LSTM + CNN )')
        self.Config_dict = Config
        self.x = self.Config_dict['input']
        self.y = self.Config_dict['output']
        self.tx = self.Config_dict['input_test']
        self.ty = self.Config_dict['output_test']
        if self.Config_dict['load_file'] != '':
            self.isPreTrained = True
```

This is the constructor to our model. It takes a dictionary as input to determine some of the fields to be used in the implementation of the networks. Next, we implement the function to build the network

```python
def constructFunctional(self):
    if self.isPreTrained:
        print('\nBuilding model...')
        self.funcModel = load_model(self.Config_dict['load_file'])
        print('\nBuilding complete!')
        self.funcModel.summary()
    else:
        print('\nBuilding model...')
        text_input = Input((self.Config_dict['input_dim'],),name = 'text_input')
        text_embed = Embedding(self.Config_dict['vocabulary_dimension'],self.Config_dict['text_embedding_dimension'],name = 'text_embedding',mask_zero = True)(text_input)

        text_model = LSTM(self.Config_dict['hidden_state_dimension'],
                      return_sequences=False,name ='lstm_layer2')(text_embed)
        vision_model = Sequential()
        #vision_model.name='CNN layer'
        vision_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=self.Config_dict['visual_dim']))

        vision_model.add(MaxPooling2D((2, 2)))
        vision_model.add(Dropout(0.5))
        vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

        vision_model.add(MaxPooling2D((2, 2)))
        vision_model.add(Dropout(0.5))
        vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

        vision_model.add(MaxPooling2D((2, 2)))
        vision_model.add(Dropout(0.5))

        vision_model.add(Flatten())

        vision_model.add(Dense(self.Config_dict['img_embedding_dimension'],activation = 'relu'))

        encoded_image = vision_model(image_input)

        merged = keras.layers.multiply([text_model,encoded_image],name= 'concat_layer')

        output_layer = Dense(self.Config_dict['output_dimension'], activation = 'softmax',name = 'output_layer')(merged)
        self.funcModel = Model(inputs=[text_input,image_input], outputs=output_layer)
        self.funcModel.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        print('\nBuilding complete!')
        self.funcModel.summary()

```

This function utilizes the _Functional_ API of Keras, because in Merging models, _Sequential_ APi can cause errors. The last few lines in the code, show the multiplication operation. That is, we get the predictions from both models individually, and calculate a dot product of them. Then, we use another ANN(Feed-Forward) with 1-hidden layer to train on these dot-product predictions, and give out the resulting answer that was predicted within the bounds of the vocabulary.

Finally, we implement the operators that will be used for the training or prediction steps:

```python
    def run(self):
        if not self.isPreTrained:
            print('\nStart Training...\n')
            #checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc',save_best_only=True, verbose=2)
            #early_stopping = EarlyStopping(monitor="val_loss", patience=5)
            #Fit the model to the data - Batch size and epoch count is configurable
            checkpoint = ModelCheckpoint(filepath = self.Config_dict['filename'],monitor='val_acc'
                                        ,verbose=1,save_best_only= True)
            earlystop = EarlyStopping(monitor='val_acc', min_delta=0,
                                        patience=5, mode='auto', restore_best_weights=True)
            #tensorboard = TensorBoard(log_dir=self.Config_dict['logdir'], histogram_freq=0,
            #                            write_graph=True, write_images=True)
            self.funcModel.fit(self.x, self.y,
              batch_size=self.Config_dict['batch_size'],
              epochs=self.Config_dict['epoch_count'],
              validation_split=0.15,callbacks = [ checkpoint,earlystop])

    def evaluate(self):
        print('\nEvaluating the accuracy...\n')
        #Evaluate the accuracy on test set
        loss, acc = self.funcModel.evaluate(self.tx, self.ty,
                           batch_size=self.Config_dict['batch_size'])
        print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    def predict(self):
        print('\nPredict Results...\n')
        #Predict answers
        ans = self.funcModel.predict(self.tx, batch_size=512)
        #print(ans)
        return np.argmax(ans, axis = -1)
```

Note that, all of these functions will reside in the class definition.
## Accuracy Measures & Training the Network <a name="wups">

In the context of NLP and other text processing problems, using a regular accuracy metric would not be the best practice. The reason is that, we would need to check if our answers are semantically correct. For example, if the question is "How many chairs are around the table ?" , we would want to see a numeric answer such as "four/4". However, if the network answers this question with a meaningless prediction, e.g. "red" , then the training process went wrong, and our network practically has not learned the latent structure of the problem.


Reflecting to this point, we will be using the WUPS(Wu-Palmer similarity) measure in this project to evaluate the results. Therefore, we will write helpre functions that can calculate wups score between two words.

```python
#Decode the one-hot into words
def decodeAnswer(answer_list,vocab):
    decoded = []
    for ans in answer_list:
        dcd = list(vocab.keys())[list(vocab.values()).index(ans)]
        decoded.append(dcd)
    return decoded

#Calculate WUPS for two words, with a certain threshold
def wup_measure(word1,word2,similarity_threshold=0.9):
    word1_sem = wn.synsets(word1,pos=wn.NOUN)
    word2_sem = wn.synsets(word2,pos=wn.NOUN)
    score=0.0

    #Calculate similarity for semantic fields
    for x in word1_sem:
        for y in word2_sem:
            local_score=x.wup_similarity(y)
            if local_score > score:
                score=local_score

    #Formula for thresholded WUPS
    if score < similarity_threshold:
        weight = 0.1
    else:
        weight = 1.0

    final_score = weight * score
    return final_score

#Computer WUPS over the whole set of answers
def computeWups(answer_list,truth):
    score = 0
    for i in range(len(answer_list)):
        score += wup_measure(answer_list[i],truth[i])
    return (score / float(len(answer_list)))
```

In order to do this, we will first convert the one-hot vector of prediction into the corresponding word from our vocabulary. Then, we will calculate the wups score between our prediction and true label.


### Training the Network

Now, we can write the main() method that will run upon our call of the function from the cmd


```python
#Main function
def main():
    print('\n Load the questions...')
    #Get the processed data for training & testing
    img_labels_train,train_questions,train_answers,vocabulary_que,vocabulary_ans = run(train_data)
    img_labels_test,test_questions,test_answers,vocabulary_que_test,vocabulary_ans_test = run(test_data,vocab = vocabulary_que, vocab_ans = vocabulary_ans)
    print('\n Load the images...')
    #Get image features
    jsd = readJsonData('img_features.json')
    img_features = createDatasetFromJson(jsd)
    img_features_train = duplicateImages(img_labels_train,img_features)
    img_features_test = duplicateImages(img_labels_test,img_features)

    #Concat the input
    extended_input_train = [train_questions,img_features_train]
    extended_input_test = [test_questions,img_features_test]

    #Construct the dictionary that inputs the configuration for the LSTM
    Config_dict = {
    'input' : extended_input_train,
    'output' : train_answers,
    'input_test' :extended_input_test,
    'output_test' : test_answers,
    'merge_mode': 'sum',
    'input_dim' : train_questions.shape[1],
    'vocabulary_dimension' : len(vocabulary_que.keys()),#train_questions.shape[1],
    'output_dimension' : len(vocabulary_ans.keys()),#train_answers.shape[1],
    'visual_dim':img_features_train.shape[1:],
    'text_embedding_dimension' : 760,
    'img_embedding_dimension' : 760,
    'hidden_state_dimension' : 760,
    'epoch_count' : 20,
    'batch_size' : 512,
    'filename' : 'Path-to-save-trained-model Example : /Documents/trained-model.h5',
    'load_file': ''
    }

    #Create,compile,train and evaluate the model
    rnnModel = MergedModel(Config_dict)
    rnnModel.constructFunctional()
    rnnModel.run()
    rnnModel.evaluate()
    #Get the predictions of LSTM
    ans = rnnModel.predict()
    #Decode predictions and truth labels from integers to words using the vocabulary
    dcd = decodeAnswer(ans,vocabulary_ans)
    dcd_truth = decodeAnswer(np.argmax(test_answers,axis = -1),vocabulary_ans)
    #Compute WUPS for all predictions
    score = computeWups(dcd,dcd_truth)
    print('WUPS Test:')
    print('Accuracy with WUPS is :{:.4f}'.format(score))
    return dcd
    #print(confusion_matrix(np.argmax(test_answers,axis = -1) , ans))

if __name__ == '__main__':
    train_data = 'qa-train.txt'
    test_data = 'qa-test-try.txt'
    dcd = main()
```


## Results <a name="results">

When we start the training process, we will see the summary of our network, which is a big network with 9 million! parameters. Therefore, it can take a while for the training to finish.

![Network Summary](../assets/images/Post-assets/vqa-images/model-summary.jpg)

However, when we finish training, we can see the results, that our model has a 20% prediction accuracy, whereas its WUPS score is 0.2352

Although these results might seem to be very low, the inherent difficulty of the task hinders us to reach very high levels of accuracy. Moreover, when compared to the results in [[Malinowski et. al. ,2015]](https://arxiv.org/abs/1505.01121), we see that our results are similar to what the authors have achieved. So we can at least confirm that our implementation was correct.


## Discussion

In this project, we have implemented a multimodal model from scratch and tested it using a visual question-answer dataset. Multimodal models are useful in many fields of commerce, one of them is information retrieval. Moreover, although we have merged an image and text based learning approaches, one can merge different channels of information to deal with many complicated tasks.

Since these models are more relaxed in architecture, but more complex in structure, the results achieved might not be of state-of-art performances. However, one can always seek for improvement in that sense...

For the resources, you can check the GitHub repository below and download the training dataset/code


[Github](https://github.com/koralpc/visual-qa)

[Dataset for image features](https://1drv.ms/u/s!AvLibAQs0au_h1MJg11PXGPUhfWr?e=kQt5EQ)
