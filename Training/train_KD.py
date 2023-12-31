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
import csv
import datetime
import tensorflow as tf
from lipnet.online_kd import extract_features, ensembling_strategy, compute_ensemble_output, multiloss_function

np.random.seed(55)

DATASET_DIR = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR = os.path.join(CURRENT_PATH, 'results')
LOG_DIR = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, 'dictionaries', 'phrases.txt')


def curriculum_rules(epoch):
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}


# @tf.function
def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size,
          num_samples_stats, peer_networks_n, distillation_strength, temperature):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch, is_val=True).build()

    peer_networks_list = []
    optimizers = []
    for n in range(peer_networks_n):
        lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                        absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
        lipnet.summary()

        # load weights
        if start_epoch == 0:
            start_file_w = os.path.join(OUTPUT_DIR, 'startWeight/unseen-weights178.h5')
            lipnet.model.load_weights(start_file_w)

        # load preexisting trained weights for the model weights113_peer_01.h5
        if start_epoch > 0:
            weight_file = os.path.join(OUTPUT_DIR, os.path.join(f"2023_10_31_17_06_50/s{n}", f'weights%02d_peer_0{n}.h5' % (start_epoch - 1)))
            lipnet.model.load_weights(weight_file)

        adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        peer_networks_list.append(lipnet)
        optimizers.append(adam)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    callback_list = []
    for n, lipnet in enumerate(peer_networks_list):
        # define callbacks
        statistics = Statistics(lipnet, lip_gen.next_val(), decoder, num_samples_stats,
                                output_dir=os.path.join(OUTPUT_DIR, run_name, "s{}".format(n)))
        visualize = Visualize(os.path.join(OUTPUT_DIR, run_name, "s{}".format(n)), lipnet, lip_gen.next_val(), decoder,
                              num_display_sentences=minibatch_size)

        statistics.on_train_begin()

        callback_list.append((statistics, visualize))

    # Csv file to store KD metrics
    student_losses_header = [f"Loss student {n}" for n in range(peer_networks_n)]
    header_losses_csv = ["Epoch - Batch"] + student_losses_header + ["Ensemble loss", "Multiloss value"]

    with open(os.path.join(OUTPUT_DIR, run_name, 'training_metrics.csv'), 'w') as csvfile:
        csvw = csv.writer(csvfile)
        csvw.writerow(header_losses_csv)

    # Csv file to store train/val losses
    student_losses_header = [f"{phase}_loss_s{n}" for n in range(peer_networks_n) for phase in ["train", "val"]]
    header_losses_csv = ["Epoch"] + student_losses_header

    with open(os.path.join(LOG_DIR, f'training-{run_name}.csv'), 'w') as csvfile:
        csvw = csv.writer(csvfile)
        csvw.writerow(header_losses_csv)

    # Callback to initialize information for curriculum
    lip_gen.on_train_begin()

    for epoch in range(start_epoch, stop_epoch):

        # Store losses of each peer network for each batch
        epoch_losses = []

        # callback to update curriculum rules
        lip_gen.on_epoch_begin(epoch)
        print("Epoch {}/{}".format(epoch, stop_epoch))

        # For each batch train simultaneously n students
        for batch in range(int(lip_gen.default_training_steps)):
            print("Batch {}/{}".format(batch, int(lip_gen.default_training_steps)))

            x_train, y_train = next(lip_gen.next_train())

            student_logits = []
            student_losses = []

            f1_array = np.array([])
            f2_array = np.array([])
            f3_array = np.array([])

            # train all students on the same batch
            with tf.GradientTape() as tape, tf.GradientTape() as tape1:
                n = 0
                statistics = callback_list[n][0]
                model_out = peer_networks_list[n].model(x_train, training=True)
                logits = model_out[0]
                losses = model_out[1]
                # Compute the mean CTC loss
                student_ctc_loss = tf.reduce_mean(losses, axis=0)
                print("Student {} mean loss: {}".format(n, student_ctc_loss))
                student_logits.append((n, logits))
                student_losses.append(student_ctc_loss)

                f1_array, f2_array, f3_array = extract_features(statistics, logits, x_train, f1_array, f2_array,
                                                                f3_array)

                n = 1
                statistics = callback_list[n][0]
                model_out = peer_networks_list[n].model(x_train, training=True)
                logits = model_out[0]
                losses = model_out[1]
                # Compute the mean CTC loss
                student_ctc_loss = tf.reduce_mean(losses, axis=0)
                print("Student {} mean loss: {}".format(n, student_ctc_loss))
                student_logits.append((n, logits))
                student_losses.append(student_ctc_loss)

                f1_array, f2_array, f3_array = extract_features(statistics, logits, x_train, f1_array, f2_array,
                                                                f3_array)

                # append array of losses in another array dedicated at each epoch
                epoch_losses.append(student_losses)

                # At the end of each batch compute ensemble weights
                student_weights = ensembling_strategy(f1_array.T, f2_array.T, f3_array.T, peer_networks_n)

                # Sum weighted logits to compute ensemble output
                ensemble_output = compute_ensemble_output(student_logits, student_weights)

                ensemble_loss, multiloss_value = multiloss_function(peer_networks_n, ensemble_output, x_train,
                                                                    student_logits, temperature,
                                                                    student_losses, distillation_strength)

                print("Multiloss value:  {}".format(multiloss_value))

            # Optimize students
            n = 0
            print("Optimizing student {}".format(n))
            gradients = tape.gradient(multiloss_value, peer_networks_list[n].model.trainable_variables)
            optimizers[n].apply_gradients(zip(gradients, peer_networks_list[n].model.trainable_variables))

            n = 1
            print("Optimizing student {}".format(n))
            gradients = tape1.gradient(multiloss_value, peer_networks_list[n].model.trainable_variables)
            optimizers[n].apply_gradients(zip(gradients, peer_networks_list[n].model.trainable_variables))

            # Save ctc losses and multiloss
            with open(os.path.join(OUTPUT_DIR, run_name, 'training_metrics.csv'), 'a') as csvfile:
                csvw = csv.writer(csvfile)
                csvw.writerow([f"Epoch {epoch} - Batch {batch}"] + [student_losses[n].numpy()[0] for n in
                                                                    range(peer_networks_n)] + [ensemble_loss.numpy()[0],
                                                                                               multiloss_value.numpy()[
                                                                                                   0]])

        # Save weights for each student every epoch
        for n, lipnet in enumerate(peer_networks_list):
            lipnet.model.save_weights(
                os.path.join(OUTPUT_DIR, run_name, "s{}".format(n), "weights{:02d}_peer_{:02d}.h5".format(epoch, n)))

        student_val_losses = []
        # Save statistics and decoded phrases for each student
        for callbacks in callback_list:
            statistics = callbacks[0]
            visualize = callbacks[1]

            validation_loss = statistics.on_epoch_end(epoch)  # return mean val loss
            print(f"Validation loss: {validation_loss}")

            student_val_losses.append(validation_loss)
            visualize.on_epoch_end(epoch)

        # Save logs
        # Training losses
        mean_epoch_losses = np.mean(np.array(epoch_losses), axis=0)

        # Merge Train and validation losses
        results = np.squeeze(np.stack((mean_epoch_losses, np.array(student_val_losses).reshape(-1, 1))).T)

        with open(os.path.join(LOG_DIR, f'training-{run_name}.csv'), 'a') as csvfile:
            csvw = csv.writer(csvfile)
            row = [f"Epoch {epoch}"] + [value for student_loss in results for value in student_loss]
            csvw.writerow(row)


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    start_epoch = 0
    stop_epoch = 340

    # Samples properties
    img_c = 3  # Image channels
    img_w = 100  # Image width
    img_h = 50  # Image height
    frames_n = 100  # Number of frames per video

    absolute_max_string_len = 54  # Max sentence length

    minibatch_size = 16  # Minibatch size

    num_samples_stats = 95  # Number of samples for statistics evaluation per epoch

    # KD parameters
    peer_networks_n = 2  # Number of peer networks
    distillation_strength = 1  # Distillation strength
    temperature = 3  # Temperature for softmax applied to logits

    train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size,
          num_samples_stats, peer_networks_n, distillation_strength, temperature)
