import sys, os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
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

DATASET_DIR = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR = os.path.join(CURRENT_PATH, 'results')
LOG_DIR = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, 'dictionaries', 'phrases.txt')


def curriculum_rules(epoch):
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}


def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size, num_samples_stats):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch).build()

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
    lipnet.summary()

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
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
    statistics = Statistics(lipnet, lip_gen.next_val(), decoder, num_samples_stats, output_dir=os.path.join(OUTPUT_DIR, run_name))
    visualize = Visualize(os.path.join(OUTPUT_DIR, run_name), lipnet, lip_gen.next_val(), decoder,
                          num_display_sentences=minibatch_size)
    tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, run_name))
    csv_logger = CSVLogger(os.path.join(LOG_DIR, "{}-{}.csv".format('training', run_name)), separator=',', append=True)
    checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, run_name, "weights{epoch:02d}.h5"), monitor='val_loss',
                                 save_weights_only=True, mode='auto', period=1)

    lipnet.model.fit_generator(generator=lip_gen.next_train(),
                               steps_per_epoch=lip_gen.default_training_steps, epochs=stop_epoch,
                               validation_data=lip_gen.next_val(), validation_steps=lip_gen.default_validation_steps,
                               callbacks=[checkpoint, statistics, visualize, lip_gen, tensorboard, csv_logger],
                               initial_epoch=start_epoch,
                               verbose=1,
                               workers=2,
                               use_multiprocessing=False)


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    start_epoch = 0
    stop_epoch = 100

    # Samples properties
    img_c = 3  # Image channels
    img_w = 100  # Image width
    img_h = 50  # Image height
    frames_n = 100  # Number of frames per video

    absolute_max_string_len = 54  # Max sentence length

    minibatch_size = 32  # Minibatch size

    num_samples_stats = 95  # Number of samples for statistics evaluation per epoch

    train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size,
          num_samples_stats)