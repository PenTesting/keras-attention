"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
import os
import argparse

from keras.callbacks import ModelCheckpoint
from models.NMT_wb import simpleNMT
from reader_wb import Data,Vocabulary
from utils.metrics import all_acc
#from utils.examples import run_examples

cp = ModelCheckpoint("./weights/NMT.{epoch:02d}-{val_loss:.2f}.hdf5",
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
    input_vocab = Vocabulary('./data/vocabulary_inckbtup.json', padding=args.padding)
    output_vocab = Vocabulary('./data/vocabulary_inckbtup.json',
                              padding=args.padding)
    kb_vocab=Vocabulary('./data/vocabulary_inckbtup.json',
                              padding=4)
    print('Loading datasets.')

    training = Data(args.training_data, input_vocab,output_vocab, kb_vocab)
    validation = Data(args.validation_data, input_vocab,output_vocab, kb_vocab)
    training.load()
    validation.load()
    training.transform()
    training.kb_out()
    validation.transform()
    validation.kb_out()
    print('Datasets Loaded.')
    print('Compiling Model.')
    model = simpleNMT(pad_length=args.padding,
                      batch_size=args.batch_size,
                      n_chars=input_vocab.size(),
                      n_labels=1523,
                      embedding_learnable=True,
                      encoder_units=256,
                      decoder_units=256,
                      trainable=True,
                      return_probabilities=False)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', all_acc])
    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')

    try:
        model.fit_generator(generator=training.generator(args.batch_size),
                            steps_per_epoch=300,
                            validation_data=validation.generator(args.batch_size),
                            validation_steps=10,
                            callbacks=[cp],
                            workers=1,
                            verbose=1,
                            epochs=args.epochs)

    except KeyboardInterrupt as e:
        print('Model training stopped early.')
    model.save_weights("model_weights_p20_e500.hdf5")

    print('Model training complete.')

    #run_examples(model, input_vocab, output_vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=500, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='1', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=20, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/train_append.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/val_append.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=20, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
