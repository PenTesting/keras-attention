"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
import os
import argparse
import random

from keras.callbacks import ModelCheckpoint
from keras.models import Model

from models.NMT import simpleNMT
from data.reader import Data,Vocabulary
from utils.metrics import all_acc
#from utils.examples import run_examples

_buckets = [(5,5),(10,10),(15, 15), (20, 20), (25, 25),(40,40)]
num_epochs=10 #Iterating over the buckets for better efficiency

cp = ModelCheckpoint("./weights/weights.hdf5",
                     monitor='val_loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto')

# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Dataset functions
    input_vocab = Vocabulary('./data/vocabulary_drive.json', padding=args.padding)
    output_vocab = Vocabulary('./data/vocabulary_drive.json',
                              padding=args.padding)

    print('Loading datasets.')

    training = Data(args.training_data, input_vocab, output_vocab)
    validation = Data(args.validation_data, input_vocab, output_vocab)
    training.load()
    validation.load()
    print('Datasets Loaded.')
    print('Compiling Model.')

    #for i in range(num_epochs):
    for bucket_id,_ in enumerate(_buckets):
        training.transform(bucket_id)
        validation.transform(bucket_id)
        print(output_vocab.size())
        #print(input_vocab.size(), output_vocab.size())
        #print(_buckets[bucket_id][0])
        model = simpleNMT(pad_length=_buckets[bucket_id][0],
                          n_chars=input_vocab.size(),
                          n_labels=output_vocab.size(),
                          embedding_learnable=False,
                          encoder_units=256,
                          decoder_units=256,
                          trainable=True,
                          return_probabilities=False)

        #model=Model(inputs=inputs,outputs=outputs)
        #print(inputs.shape,outputs.shape)
        model.summary()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', all_acc])
        '''if os.path.exists("./weights/weights.hdf5"):
            model.load_weights("./weights/weights.hdf5")
            print("Model Weigths loaded successfully")'''
        print('Model Compiled.')
        print('Training. Ctrl+C to end early.')

        try:
            model.fit_generator(generator=training.generator(args.batch_size,bucket_id),
                                steps_per_epoch=100,
                                validation_data=validation.generator(args.batch_size,bucket_id),
                                validation_steps=5,
                                callbacks=[cp],
                                workers=1,
                                verbose=1,
                                epochs=args.epochs)

        except KeyboardInterrupt as e:
            print('Model training stopped early.')
        model.save_weights("weights"+str(bucket_id)+".hdf5")
        print('Saving Model')
        model.save("model"+str(bucket_id)+".hdf5")
        print("Model saved")

    print('Model training complete.')
    
    #run_examples(model, input_vocab, output_vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=40, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='1', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=40, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/training_drive.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/validation_drive.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=50, type=int)
    args = parser.parse_args()
    print(args)

    main(args)
