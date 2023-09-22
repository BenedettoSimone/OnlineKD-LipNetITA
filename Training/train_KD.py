import sys, os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from lipnet.generators import BasicGenerator
from lipnet.callbacks import Statistics, Visualize
from lipnet.curriculums import Curriculum
from lipnet.decoders import Decoder
from lipnet.helpers import labels_to_text
from lipnet.spell import Spell
from lipnet.model import LipNet
import numpy as np
import datetime

np.random.seed(55)

DATASET_DIR  = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR   = os.path.join(CURRENT_PATH, 'results')
LOG_DIR      = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'dictionaries','phrases.txt')

def curriculum_rules(epoch):
    return { 'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05 }

def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size, peer_networks_n):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch, is_val=True).build()

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
    lipnet.summary()

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    # load weights
    if start_epoch == 0:
        start_file_w = os.path.join(OUTPUT_DIR, 'startWeight/unseen-weights178.h5')
        lipnet.model.load_weights(start_file_w)

    # load preexisting trained weights for the model
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        lipnet.model.load_weights(weight_file)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    # define callbacks
    statistics = Statistics(lipnet, lip_gen.next_val(), decoder, 95, output_dir=os.path.join(OUTPUT_DIR, run_name))
    visualize = Visualize(os.path.join(OUTPUT_DIR, run_name), lipnet, lip_gen.next_val(), decoder,
                          num_display_sentences=minibatch_size)

    # Training
    for epoch in range(start_epoch, stop_epoch):
        print("Epoch {}/{}".format(epoch + 1, stop_epoch))

        n = 0
        # int(lip_gen.default_training_steps)
        for batch in range(1):
            print("Batch {}/{}".format(n, int(lip_gen.default_training_steps)))
            x_train, y_train = next(lip_gen.next_train())
            loss = lipnet.model.train_on_batch(x_train, y_train)
            n = n+1

        val_loss = 0.0
        num_val_batches = int(lip_gen.default_validation_steps)
        for batch in range(1):
            x_val, y_val = next(lip_gen.next_val())
            val_loss += lipnet.model.evaluate(x_val, y_val, verbose=0)

        val_loss /= num_val_batches
        print("Epoch {} - Loss: {} - Val Loss: {}".format(epoch + 1, loss, val_loss))

        lipnet.model.save_weights(os.path.join(OUTPUT_DIR, run_name, "weights{:02d}_peer_{:02d}.h5".format(epoch, 1)))
        statistics.on_epoch_end(epoch)
        visualize.on_epoch_end(epoch)



if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # 1st parameter - run_name
    # 2nd parameter - start_epoch
    # 3rd parameter - stop_epoch
    # 4th parameter - img_c (num of channel)
    # 5th parameter - img_w
    # 6th parameter - img_h
    # 7th parameter - frames_n
    # 8th parameter - absolute_max_string_length (max len of sentences)
    # 9th parameter - minibatch_size
    # 10th parameter - number of peer network
    train(run_name, 0, 10, 3, 100, 50, 100, 54, 19, 3)
