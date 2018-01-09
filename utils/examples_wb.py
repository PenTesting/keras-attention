import numpy as np
from models.NMT import simpleNMT
from keras.models import Model
from reader_wb import Data,Vocabulary
import pandas as pd

#EXAMPLES = ['26th January 2016', '3 April 1989', '5 Dec 09', 'Sat 8 Jun 2017']
_buckets = [(5,5),(10,10),(15, 15), (20, 20), (25, 25),(40,40)]
EXAMPLES=["find starbucks","What will the weather in Fresno be in the next 48 hours","give me directions to the closest grocery store","What is the address?","Remind me to take pills","tomorrow in inglewood will it be windy?"]
def run_example(model, input_vocabulary, output_vocabulary, text):
    for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(text.split(" ")) < source_size :
            padding=_buckets[bucket_id][0]
            break
    encoded = input_vocabulary.string_to_int(text)
    #print(encoded.shape)
    prediction = model.predict(np.array([encoded]))
    #print(prediction, type(prediction), prediction.shape)
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
if __name__=="__main__":
    df=pd.read_csv("../data/validation_complete.csv")
    inputs=list(df["inputs"])
    outputs=list(df["outputs"])
    input_vocab = Vocabulary('../data/vocabulary_drive.json', padding=40)
    output_vocab = Vocabulary('../data/vocabulary_drive.json',padding=40)
    
    print(inputs[0],outputs[0],len(inputs),len(outputs),type(inputs))
    model=simpleNMT(pad_length=40,
                      n_chars=input_vocab.size(),
                      n_labels=output_vocab.size(),
                      embedding_learnable=False,
                      encoder_units=256,
                      decoder_units=256,
                      trainable=True,
                      return_probabilities=False)
    model.summary()
    weights_file="../weights/NMT.19-1.18.hdf5"
    model.load_weights(weights_file, by_name=True)
    run_examples(model,input_vocab,output_vocab,inputs[:5])
    print(outputs[:5])

