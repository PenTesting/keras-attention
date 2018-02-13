import numpy as np
import pandas as pd
from models.NMT import simpleNMT
from keras.models import Model
from data.reader import Data,Vocabulary

#EXAMPLES = ['26th January 2016', '3 April 1989', '5 Dec 09', 'Sat 8 Jun 2017']
_buckets = [(5,5),(10,10),(15, 15), (20, 20), (25, 25),(40,40)]
EXAMPLES=["find starbucks","What will the weather in Fresno be in the next 48 hours","give me directions to the closest grocery store","What is the address?","Remind me to take pills","tomorrow in inglewood will it be windy?"]
def run_example(model, input_vocabulary, output_vocabulary, text):
    for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(text.split(" ")) < source_size :
            padding=_buckets[bucket_id][0]
            break
    encoded = input_vocabulary.string_to_int(text,padding)
    print(encoded.shape)
    prediction = model.predict(np.array([encoded]))
    print(prediction, type(prediction), prediction.shape)
    prediction = np.argmax(prediction[0], axis=-1)
    return output_vocabulary.int_to_string(prediction)

def run_examples(model, input_vocabulary, output_vocabulary, examples=EXAMPLES):
    predicted = []
    for example in examples:
        print('~~~~~')
        predicted.append(''.join(run_example(model, input_vocabulary, output_vocabulary, example)))
        print('input:',example)
        print('output:',predicted[-1])
    return predicted


def create_testset():
    #f = open("output_withoutkb.txt", "a")
    df=pd.read_csv('../data/testing_complete.csv')
    input=list(df["inputs"])
    output=list(df["outputs"])
    d = {'outputs': [], 'inputs': [], 'prediction': []}
    for bucket_id, (source_size, target_size) in enumerate(_buckets):
        weights_file = "../weights" + str(bucket_id) + ".hdf5"
        model = simpleNMT(pad_length=_buckets[bucket_id][0],
                          n_chars=1523,
                          n_labels=1523,
                          embedding_learnable=False,
                          encoder_units=256,
                          decoder_units=256,
                          trainable=False,
                          return_probabilities=False)
        for text, op in zip(input, output):
            if len(text.split(" ")) < source_size:
                #model = Model(inputs=inputs, outputs=outputs)
                #print(outputs.shape)
                input_vocab = Vocabulary('../data/vocabulary_drive.json', padding=None)
                output_vocab = Vocabulary('../data/vocabulary_drive.json', padding=None)
                #print(input_vocab.size(), output_vocab.size())
                model.load_weights(weights_file, by_name=True)
                model.compile(optimizer='adam', loss='categorical_crossentropy')
                # run_examples(model, input_vocab, output_vocab)
                encoded = input_vocab.string_to_int(text,_buckets[bucket_id][0])
                #print(encoded, np.array([encoded]).shape)
                prediction = model.predict(np.array([encoded]))
                # prediction=prediction.reshape((1,10,10))
                #print(prediction.shape, prediction)
                prediction = np.argmax(prediction[0], axis=-1)
                #print(text, output_vocab.int_to_string(prediction))
                d["outputs"].append(str(op))
                d["inputs"].append(str(text))
                d["prediction"].append(str(output_vocab.int_to_string(prediction)))
                #f.write(str(text) + "," + str(op) + ","+str(output_vocab.int_to_string(prediction))+"\n")
    #f.close()
    df = pd.DataFrame(d)
    df.to_csv("output_withoutkb.csv")


create_testset()
