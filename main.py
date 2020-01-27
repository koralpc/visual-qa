#Author : Koralp Catalsakal
#Project : TextBasedIR Project Part 1
#Date : 23/03/2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolz import frequencies
import operator
import string
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from sklearn.metrics import confusion_matrix
from nltk.corpus import wordnet as wn
import nltk
import sys
import json
from sklearn.model_selection import train_test_split
#Update the wordnet package
nltk.download('wordnet')

#Get the file directory
#if len(sys.argv) >= 2:
#print('\nCustom path selected!')
#train_data = sys.argv[len(sys.argv) -2]
#test_data = sys.argv[ len(sys.argv) -1]
#else:
#print('\nDefault path selected! ')
train_data = 'qa-train.txt'
test_data = 'qa-test.txt'

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

#Create vocabulary from the word counts - for answers
def createVocabularyAnswers(freqs):
    vocabulary = [('<pad>',0),('<unk>', 1),('<eoa>', 2),('<eoq>', 3)]
    sorted_freqs = sorted(freqs.items(), key=operator.itemgetter(1) , reverse = True)
    j = 4
    for i,v in enumerate(sorted_freqs):
        vocabulary.append((sorted_freqs[i][0],j))
        j += 1
    return dict(vocabulary)

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

#Sequence padding to have a uniform length over all inputs
def sequencePad(data,MAXLEN = 30):
    padded = sequence.pad_sequences(data, maxlen=MAXLEN)
    return padded

#Duplicate the used images for part2
def duplicateImages(label_array,img_array):
    retArr = []
    for i in range(0,len(label_array)):
        #Minus one because the first index of an array is 0
        index = int(label_array.iat[i,0].replace('image','')) - 1
        retArr.append(img_array[index])
    #retArr2 = np.array(retArr)
    return np.array(retArr)

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
        vocabulary_ans = createVocabularyAnswers(freqs_ans)
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

#The RNNModel that is used in the project
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

    #Create the LSTM model and compile
    def construct(self):
        print('\nBuilding model...')
        #Add word embedding to the LSTM with the given embedding dimension as a tunable parameter
        self.rnnModel.add(Embedding(
                self.Config_dict['input_dimension'],
                self.Config_dict['text_embedding_dimension'],
                mask_zero=True))
        #Add the LSTM layer to the model with the number of hidden neurons
        self.rnnModel.add(LSTM(self.Config_dict['hidden_state_dimension'],
                      return_sequences=False))
        #Add a layer for visual processing. Since the features are already extracted, no need for complex networks.
        self.visualModel.add(Dense(self.Config_dict['img_embedding_dimension'],input_shape=(self.Config_dict['visual_dim'],)))
        #Add the merge layer to merge two inputs
        self.genericModel.add(Multiply([self.rnnModel, self.visualModel]))
        #Add optional dropout, for protection against local minimas
        self.genericModel.add(Dropout(0.5))
        #Add the Dense layer as the output layer
        self.genericModel.add(Dense(self.Config_dict['output_dimension']))
        #The task is categorical, so the output activation function is softmax
        self.genericModel.add(Activation('softmax'))
        #Compile with categorical_crossentropy loss and using 'adam' optimization
        self.genericModel.compile(loss='categorical_crossentropy',optimizer='adam')
        print('\nBuilding complete!')
        #self.genericModel.summary()

    def constructFunctional(self):
        if self.isPreTrained:
            print('\nBuilding model...')
            self.funcModel = load_model(self.Config_dict['load_file'])
            print('\nBuilding complete!')
            self.funcModel.summary()
        else:
            #first_embed = Input(
            #        (self.Config_dict['input_dimension'],
            #        self.Config_dict['text_embedding_dimension']))
            print('\nBuilding model...')
            text_input = Input((self.Config_dict['input_dim'],),name = 'text_input')
            text_embed = Embedding(self.Config_dict['vocabulary_dimension'],self.Config_dict['text_embedding_dimension'],name = 'text_embedding',mask_zero = True)(text_input)
            #first_model = LSTM(self.Config_dict['hidden_state_dimension'],
            #              return_sequences=True,name ='lstm_layer')(text_embed)
            #first_model = Dropout(0.5)(first_model)
            text_model = LSTM(self.Config_dict['hidden_state_dimension'],
                          return_sequences=False,name ='lstm_layer2')(text_embed)
            #first_model = Dropout(0.5)(first_model)
            #first_model = Dense(self.Config_dict['hidden_state_dimension'],activation='tanh')(first_model)

            vision_model = Sequential()
            #vision_model.name='CNN layer'
            vision_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=self.Config_dict['visual_dim']))
            #vision_model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
            #vision_model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
            #vision_model.add(BatchNormalization())
            vision_model.add(MaxPooling2D((2, 2)))
            vision_model.add(Dropout(0.5))
            vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            #vision_model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
            #vision_model.add(BatchNormalization())
            vision_model.add(MaxPooling2D((2, 2)))
            vision_model.add(Dropout(0.5))
            vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    ##        vision_model.add(BatchNormalization())
            vision_model.add(MaxPooling2D((2, 2)))
            vision_model.add(Dropout(0.5))
            #vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            vision_model.add(Flatten())
            #vision_model.add(Dropout(0.25))
            #vision_model.add(Dense(4096,activation = 'tanh'))
            vision_model.add(Dense(self.Config_dict['img_embedding_dimension'],activation = 'relu'))
            #vision_model.add(Activation('relu'))

            image_input = Input(shape = self.Config_dict['visual_dim'],name = 'visual_input')
            #flat = Flatten()(image_input)
            #vision_model2 = Dense(self.Config_dict['img_embedding_dimension'],activation = 'tanh')(flat)
            #act = Activation('relu')(vision_model2)
            encoded_image = vision_model(image_input)

            merged = tf.keras.layers.multiply([text_model,encoded_image],name= 'concat_layer')
            #merged = Dropout(0.5)(merged)
            #merged = Dense(self.Config_dict['hidden_state_dimension'],activation='tanh')(merged)
            #merged = Dropout(0.5)(merged)
            output_layer = Dense(self.Config_dict['output_dimension'], activation = 'softmax',name = 'output_layer')(merged)
            self.funcModel = Model(inputs=[text_input,image_input], outputs=output_layer)
            self.funcModel.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
            print('\nBuilding complete!')
            self.funcModel.summary()

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
    'filename' : 'C:\\Users\\koral\\Google Drive\\KUL\\1-2\\TextBasedIR\\Project\\part1\\github-model.h5',
    #
    'load_file': 'C:\\Users\\koral\\Google Drive\\KUL\\1-2\\TextBasedIR\\Project\\part1\\deep-lstm-embed.h5'
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
    dcd
    dcd_truth = decodeAnswer(np.argmax(test_answers,axis = -1),vocabulary_ans)
    #Compute WUPS for all predictions
    score = computeWups(dcd,dcd_truth)
    print('WUPS Test:')
    print('Accuracy with WUPS is :{:.4f}'.format(score))
    return dcd
    #print(confusion_matrix(np.argmax(test_answers,axis = -1) , ans))

if __name__ == '__main__':
    train_data = 'qa-train.txt'
    test_data = 'qa-test.txt'
    dcd = main()
