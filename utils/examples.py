import numpy as np
from models.NMT import simpleNMT
from keras.models import Model
from data.reader import Data,Vocabulary

#EXAMPLES = ['26th January 2016', '3 April 1989', '5 Dec 09', 'Sat 8 Jun 2017']
_buckets = [(5,5),(10,10),(15, 15), (20, 20), (25, 25),(40,40)]
EXAMPLES=["find starbucks","What is the address?","Remind me to take pills","tomorrow in inglewood will it be windy?"]
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

weights_file="../weights1.hdf5"
inputs,outputs=simpleNMT(pad_length=10,
                          n_chars=1522,
                          n_labels=1522,
                          embedding_learnable=False,
                          encoder_units=256,
                          decoder_units=256,
                          trainable=False,
                          return_probabilities=False)

'''inputs,outputs = simpleNMT( pad_length=5,
                            n_chars=1522,
                            n_labels=1522,
                            return_probabilities=True)'''

model=Model(inputs=inputs,outputs=outputs)
print(outputs.shape)
input_vocab = Vocabulary('../data/vocabulary_drive.json', padding=None)
output_vocab = Vocabulary('../data/vocabulary_drive.json',padding=None)
print(input_vocab.size(),output_vocab.size())
model.load_weights(weights_file, by_name=True)
model.compile(optimizer='adam', loss='categorical_crossentropy')
#run_examples(model, input_vocab, output_vocab)
encoded = input_vocab.string_to_int(EXAMPLES[1],10)
print(encoded,np.array([encoded]).shape)
prediction = model.predict(np.array([encoded]))
#prediction=prediction.reshape((1,10,10))
print(prediction.shape,prediction)
prediction = np.argmax(prediction[0], axis=-1)
print(EXAMPLES[1],output_vocab.int_to_string(prediction))