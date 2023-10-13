import sys, os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from lipnet.generators import BasicGenerator
from lipnet.callbacks import Statistics, Visualize
from lipnet.curriculums import Curriculum
from lipnet.decoders import Decoder
from lipnet.helpers import labels_to_text
from lipnet.spell import Spell
from lipnet.model import LipNet
import numpy as np
import datetime
import tensorflow as tf
from lipnet.online_kd import ensembling_strategy, compute_ensemble_output, compute_ensemble_mean_loss


np.random.seed(55)

DATASET_DIR = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR = os.path.join(CURRENT_PATH, 'results')
LOG_DIR = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, 'dictionaries', 'phrases.txt')


def curriculum_rules(epoch):
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}

def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size,
          num_samples_stats, peer_networks_n):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch, is_val=True).build()

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    peer_networks_list = []

    for n in range(peer_networks_n):
        lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                        absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
        lipnet.summary()

        # load weights
        if start_epoch == 0:
            start_file_w = os.path.join(OUTPUT_DIR, 'startWeight/unseen-weights178.h5')
            lipnet.model.load_weights(start_file_w)

        # load preexisting trained weights for the model
        if start_epoch > 0:
            weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
            lipnet.model.load_weights(weight_file)

        peer_networks_list.append(lipnet)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    # Training
    for epoch in range(start_epoch, stop_epoch):

        print("Epoch {}/{}".format(epoch + 1, stop_epoch))

        # For each batch we train simultaneously n students
        b = 0

        student_predictions = []
        student_losses = []

        # TODO set int(lip_gen.default_training_steps)
        for batch in range(1):
            print("Batch {}/{}".format(b, int(lip_gen.default_training_steps)))

            x_train, y_train = next(lip_gen.next_train())

            f1_array = np.array([])
            f2_array = np.array([])
            f3_array = np.array([])

            # train all students on the same batch
            for n, lipnet in enumerate(peer_networks_list):

                # define callbacks
                statistics = Statistics(lipnet, lip_gen.next_val(), decoder, num_samples_stats,
                                        output_dir=os.path.join(OUTPUT_DIR, run_name))
                visualize = Visualize(os.path.join(OUTPUT_DIR, run_name), lipnet, lip_gen.next_val(), decoder,
                                      num_display_sentences=minibatch_size)


                with tf.GradientTape() as tape:

                    # Forward pass
                    # this result is a set of prob + losses
                    y_pred = lipnet.model(x_train, training=True)

                    # Compute the mean CTC loss
                    student_ctc_loss = tf.reduce_mean(y_pred[1], axis=0)
                    print("Student {} mean loss: {}".format(n, student_ctc_loss))

                student_predictions.append((n, y_pred[0]))
                student_losses.append(student_ctc_loss)

                '''
                ########################################################################################
                ### STEP 2: compute features
                ########################################################################################
                '''
                # make prediction on current batch, decode results and save them to compute
                # metrics useful to perform KD attention

                f1, f2, f3 = statistics.on_batch_end(x_train)

                # concatenate feature of each student for each sample
                if f1_array.size > 0:
                    f1_array = np.vstack((f1_array,f1)).T # (b_size, n_students) es. [[42.  9.], [16.  6.],...., [15.  5.]]
                else:
                    f1_array= np.append(f1_array, np.array(f1))

                if f2_array.size > 0:
                    f2_array = np.vstack((f2_array, f2)).T
                else:
                    f2_array = np.append(f2_array, np.array(f2))

                if f3_array.size > 0:
                    f3_array = np.vstack((f3_array, f3)).T
                else:
                    f3_array = np.append(f3_array, np.array(f3))

            # At the end of each batch compute ensemble weights
            student_weights = ensembling_strategy(f1_array, f2_array, f3_array, peer_networks_n)

            # Sum weighted predictions to compute ensemble output
            ensemble_output = compute_ensemble_output(student_predictions, student_weights)

            '''
            ########################################################################################
            ### STEP 5: compute multi-loss function
            ########################################################################################
            '''
            # L = CTC loss ensemble predictions and truth + sum(1,N)[(CTC loss student predictions(i) and truth)+
            # (LDK divergence students predictions(i) and ensemble predictions)

            # Ensemble CTC loss
            ensemble_mean_loss = compute_ensemble_mean_loss(ensemble_output, x_train)
            print("Ensemble mean loss: {}".format(ensemble_mean_loss))

            # CTC loss between students predictions and truth
            # Already in student_losses array

            #TODO show distribution over sequence samples of KL divergence to understand if we can choose the mean kl


            '''
            ########################################################################################
            ### STEP 5: optimize students
            ########################################################################################
            '''
            #grads = tape.gradient(mean_ctc_loss, lipnet.model.trainable_variables)
            #adam.apply_gradients(zip(grads, lipnet.model.trainable_variables))


            b = b + 1

        # Save model weights every 5 epochs
        #if (epoch + 1) % 5 == 0:
         #   for n, lipnet in enumerate(peer_networks_list):
          #      lipnet.model.save_weights(
           #         os.path.join(OUTPUT_DIR, run_name, "weights{:02d}_peer_{:02d}.h5".format(epoch, n)))

        # statistics.on_epoch_end(epoch)
        # visualize.on_epoch_end(epoch)


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
    # 10th parameter - num_samples_stats (number of samples for statistics evaluation)
    # 11th parameter - number of peer network
    train(run_name, 0, 10, 3, 100, 50, 100, 54, 19, 95, 2)
