import json
import csv
import random

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize


random.seed(1984)

INPUT_PADDING = 50
OUTPUT_PADDING = 100
_buckets = [(5,5),(10,10),(15, 15), (20, 20), (25, 25),(40,40)]

class Vocabulary(object):

    def __init__(self, vocabulary_file, padding=None):
        """
            Creates a vocabulary from a file
            :param vocabulary_file: the path to the vocabulary
        """
        self.vocabulary_file = vocabulary_file
        with open(vocabulary_file, 'r',encoding='utf-8') as f:
            self.vocabulary = json.load(f)

        #self.padding = padding
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def size(self):
        """
            Gets the size of the vocabulary
        """
        return len(self.vocabulary.keys())

    def string_to_int(self, text,padding):
        """
            Converts a string into it's character integer 
            representation
            :param text: text to convert
        """
        self.padding=padding
        tokens = word_tokenize(text)

        integers = []

        if self.padding and len(tokens) >= self.padding:
            # truncate if too long
            tokens = tokens[:self.padding - 1]

        tokens.append('<eos>')

        for c in tokens:
            if c in self.vocabulary:
                integers.append(self.vocabulary[c])
            else:
                integers.append(self.vocabulary['<unk>'])


        # pad:
        if self.padding and len(integers) < self.padding:
            integers.reverse()
            integers.extend([self.vocabulary['<unk>']]
                            * (self.padding - len(integers)))
            integers.reverse()

        if len(integers) != self.padding and self.padding:
            print(text)
            raise AttributeError('Length of text was not padding.')
        return integers

    def int_to_string(self, integers):
        """
            Decodes a list of integers
            into it's string representation
        """
        tokens = []
        for i in integers:
            tokens.append(self.reverse_vocabulary[i])

        return tokens


class Data(object):

    def __init__(self, file_name, input_vocabulary, output_vocabulary):
        """
            Creates an object that gets data from a file
            :param file_name: name of the file to read from
            :param vocabulary: the Vocabulary object to use
            :param batch_size: the number of datapoints to return
            :param padding: the amount of padding to apply to 
                            a short string
        """

        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        self.file_name = file_name

    def load(self):
        """
            Loads data from a file
        """
        self.inputs = []
        self.targets = []
        self.data_buckets=[[] for i in range(len(_buckets))]
        with open(self.file_name, 'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.inputs.append(row[1])
                self.targets.append(row[2])
        for source_ids,target_ids in zip(self.inputs,self.targets):
            flag=0
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(source_ids.split(" ")) < source_size and len(target_ids.split(" ")) < target_size:
                    self.data_buckets[bucket_id].append([source_ids, target_ids])
                    flag=1
                    break
            if (flag == 0):
                flag = 0
                self.data_buckets[5].append([source_ids, target_ids])
        for i in range(len(_buckets)):
            for j in range(len(self.data_buckets[i])):
                self.data_buckets[i][j][0]= self.input_vocabulary.string_to_int(self.data_buckets[i][j][0],_buckets[i][0])
                self.data_buckets[i][j][1] = self.input_vocabulary.string_to_int(self.data_buckets[i][j][1],_buckets[i][0])
    def transform(self,bucket_id):
        """
            Transforms the data as necessary
        """
        # @TODO: use `pool.map_async` here?
        
        '''self.inputs = np.array(list(
            map(self.input_vocabulary.string_to_int, self.inputs)))

        self.targets = np.array(list(map(self.output_vocabulary.string_to_int, self.targets)))
        x=list(map(
                lambda x: to_categorical(
                    x,
                    num_classes=self.output_vocabulary.size()+1),
                self.targets))
        self.targets = np.array(x)'''
        self.inputs=np.array(self.data_buckets[bucket_id])[:,0]
        self.targets=np.array(self.data_buckets[bucket_id])[:,1]
        self.targets = np.array(list(map(lambda x: to_categorical(x, num_classes=self.output_vocabulary.size()), self.targets)))

        #print(self.inputs.shape,self.targets.shape)
        #assert len(self.inputs.shape) == 2, 'Inputs could not properly be encoded'
        #assert len(self.targets.shape) == 3, 'Targets could not properly be encoded'

    def generator(self, batch_size,bucket_id):
        """
            Creates a generator that can be used in `model.fit_generator()`
            Batches are generated randomly.
            :param batch_size: the number of instances to include per batch
        """

        while True:
            try:

                #print(np.array(data_buckets[bucket_id][:, 0])[batch_ids], targets.shape)
                #bucket_id=random.randint(0,len(_buckets)-1)
                #print(bucket_id,_buckets[bucket_id],len(self.data_buckets[bucket_id]))
                instance_id = range(len(self.inputs))
                batch_ids = random.sample(instance_id, batch_size)
                #targets=np.array(np.array(self.data_buckets[bucket_id])[:, 1])[batch_ids]
                #targets = np.array(list(map(lambda x: to_categorical(x,num_classes=self.output_vocabulary.size()),targets)))
                #print(np.array(np.array(self.data_buckets[bucket_id])[:,0])[batch_ids].shape,np.array(targets).shape)
                yield (np.array(self.inputs[batch_ids], dtype=int),np.array(self.targets[batch_ids]))
                #yield (np.array(np.array(self.data_buckets[bucket_id])[:,0])[batch_ids],np.array(targets))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None

if __name__ == '__main__':
    input_vocab = Vocabulary('./vocabulary_drive.json', padding=40)
    output_vocab = Vocabulary('./vocabulary_drive.json', padding=40)
    ds = Data('training_drive.csv', input_vocab, output_vocab)
    ds.load()
    ds.transform(1)
    print(ds.inputs.shape)
    print(ds.targets.shape)
    print(ds.output_vocabulary.size())
    ds.generator(32,1)
    #print(ds.data_buckets[1])
    g = ds.generator(32,1)
    print(ds.targets[0])
    #print(ds.inputs[[5,10, 12]].shape)
    #print(ds.targets[[5,10,12]].shape)
    for i in range(50):
         print((next(g)[0]))
         print((next(g)[1]))
         break
