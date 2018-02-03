import numpy as np
import keras
from models.NMT_wb import simpleNMT
from keras.models import Model,load_model
from models.custom_recurrents import AttentionDecoder
from reader_wb import Data,Vocabulary
import pandas as pd


outdf={'input':[],'output':[]}
#EXAMPLES = ['26th January 2016', '3 April 1989', '5 Dec 09', 'Sat 8 Jun 2017']
_buckets = [(5,5),(10,10),(15, 15), (20, 20), (25, 25),(40,40)]
EXAMPLES=["find starbucks <eos>","What will the weather in Fresno be in the next 48 hours <eos>","give me directions to the closest grocery store <eos>","What is the address? <eos>","Remind me to take pills","tomorrow in inglewood will it be windy?"]
def run_example(model,kbs, input_vocabulary, output_vocabulary, text):
    '''for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(text.split(" ")) < source_size :
            padding=_buckets[bucket_id][0]
            break'''
    encoded = input_vocabulary.string_to_int(text)
    print("encoded is",encoded)
    prediction = model.predict([np.array([encoded]),kbs])
    #print(prediction, type(prediction), prediction.shape)
    #print(prediction[0].shape)
    prediction = np.argmax(prediction[0], axis=-1)
    #print(prediction, type(prediction), prediction.shape)
    return output_vocabulary.int_to_string(prediction)

def run_examples(model, kbs,input_vocabulary, output_vocabulary, examples=EXAMPLES):
    predicted = []
    input=[]
    for example in examples:
        print('~~~~~')
        input.append(example)
        predicted.append(' '.join(run_example(model,kbs, input_vocabulary, output_vocabulary, example)))
        outdf['input'].append(example)
        outdf['output'].append(predicted[-1])
        print('input:',example)
        #print(type(predicted),predicted)
        print('output:',predicted[-1])
    return predicted
if __name__=="__main__":
    pad_length=20
    df=pd.read_csv("../data/testing_complete.csv")
    inputs=list(df["inputs"])
    outputs=list(df["outputs"])
    input_vocab = Vocabulary('../data/vocabulary_inckbtup.json', padding=pad_length)
    output_vocab = Vocabulary('../data/vocabulary_inckbtup.json',padding=pad_length)
    kb_vocabulary = Vocabulary('../data/vocabulary_inckbtup.json',
                               padding=4)
    #print(inputs[0],outputs[0],len(inputs),len(outputs),type(inputs))
    model=simpleNMT(pad_length=pad_length,batch_size=1,
                      n_chars=input_vocab.size(),
                      n_labels=1523,
                      embedding_learnable=False,
                      encoder_units=256,
                      decoder_units=256,
                      trainable=False,
                      return_probabilities=False)
    #model=load_model("../modelkb.hdf5",custom_objects={'AttentionDecoder': AttentionDecoder})
    model.summary()
    weights_file="../model_weights.hdf5"
    model.load_weights(weights_file, by_name=True)
    
    kbfile = "../data/normalised_kbtuples.csv"
    df = pd.read_csv(kbfile)
    kbs = list(df["subject"] + " " + df["relation"])
    kbs = np.array(list(map(kb_vocabulary.string_to_int, kbs)))
    kbs=np.repeat(kbs[np.newaxis, :, :], 1, axis=0)
    data=run_examples(model,kbs,input_vocab,output_vocab,inputs)
    f=open("output.txt","w")
    for i,o in zip(outputs,data):
        f.write(str(i)+","+str(o)+"\n")
    #print(outputs)

