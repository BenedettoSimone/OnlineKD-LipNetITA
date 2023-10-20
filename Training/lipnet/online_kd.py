import tensorflow as tf
from shared_mlp import SharedMLP
import numpy as np
from layers import ctc_lambda_func
import matplotlib.pyplot as plt

def extract_features(statistics, x_train, f1_array, f2_array, f3_array):

    # Extract features
    f1, f2, f3 = statistics.on_batch_end(x_train)

    # concatenate feature of each student for each sample
    if f1_array.size > 0:
        f1_array = np.vstack(
            (f1_array, f1)).T  # (b_size, n_students) es. [[42.  9.], [16.  6.],...., [15.  5.]]
    else:
        f1_array = np.append(f1_array, np.array(f1))

    if f2_array.size > 0:
        f2_array = np.vstack((f2_array, f2)).T
    else:
        f2_array = np.append(f2_array, np.array(f2))

    if f3_array.size > 0:
        f3_array = np.vstack((f3_array, f3)).T
    else:
        f3_array = np.append(f3_array, np.array(f3))

    return f1_array, f2_array, f3_array

def ensembling_strategy(f1, f2, f3, peer_networks_n):

    # Input dimensions
    input_dim = peer_networks_n * 3
    output_dim = peer_networks_n * 3

    # SharedMLP
    mlp_shared = SharedMLP(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)

    # Concatenate features of students
    concatenated_features = tf.concat([f1, f2, f3], axis=1) # (b_size,6)[f1_peer1, f1_peer2 f2_peer1 f2_peer2 f3_peer1 f3_peer2]

    # Forward pass
    predictions = mlp_shared(concatenated_features)  # (b_size,6) [f1_peer1, f1_peer2 f2_peer1 f2_peer2 f3_peer1 f3_peer2]

    # Split mlp output
    student_weights = []
    for i in range(peer_networks_n):
        predictions_student_i = tf.gather(predictions, [i, i+peer_networks_n, i+(peer_networks_n*2)], axis=1)
        sum_predictions_student_i = tf.reduce_sum(predictions_student_i, axis=1)
        weights_student_i = tf.math.sigmoid(sum_predictions_student_i)
        student_weights.append(weights_student_i)

    return student_weights


def compute_ensemble_output(student_predictions, student_weights):
    ensemble_output = []

    # Get how many predictions
    for i in range(student_predictions[0][1].shape[0]):
        #Get weighted prediction of sample
        weighted_predictions = []
        weights_pred_i = []

        #Get i prediction of each student
        for student_idx, pred in student_predictions:
            # Multiply each tensor for the student weight for the corresponding prediction
            weighted_predictions.append(pred[i] * student_weights[student_idx][i])
            weights_pred_i.append(student_weights[student_idx][i])

        # For each sample (i.e. pred1 of s1, pred1 of s2, pred1 of s3)
        # (pred * w1 + pred * w2 + pred * w3) / w1+w2+w3
        # Sum each student tensor
        ensemble_prediction = np.sum(weighted_predictions, axis=0)
        ensemble_prediction = ensemble_prediction / np.sum(weights_pred_i)
        ensemble_output.append(ensemble_prediction)

    ensemble_output = np.array(ensemble_output)

    return ensemble_output


def compute_ensemble_loss(ensemble_output, x_train):

    labels = tf.convert_to_tensor(x_train['the_labels'])
    label_length = tf.convert_to_tensor(x_train['label_length'].reshape(-1, 1))
    input_length = tf.convert_to_tensor(x_train['input_length'].reshape(-1, 1))
    ensemble_output = tf.convert_to_tensor(ensemble_output)
    ensemble_ctc_loss = ctc_lambda_func([ensemble_output, labels, input_length, label_length])

    # Compute the mean CTC loss
    ensemble_loss = tf.reduce_mean(ensemble_ctc_loss, axis=0)

    return ensemble_loss

def kl_divergence(student_prediction, ensemble_output):
    sequence_kl_divergence = []

    for i in range(student_prediction.shape[0]):
        p = student_prediction[i]
        q = ensemble_output[i]

        kl_value = np.sum(p * np.log(p / q))
        sequence_kl_divergence.append(kl_value)

    # TODO sum or mean
    mean_kl_divergence = np.mean(sequence_kl_divergence)
    return sequence_kl_divergence, mean_kl_divergence


def kd_loss(student_predictions, ensemble_output, temperature):

    students_kl = []

    # Get i prediction of each student and compute kl divergence
    for student_idx, pred in student_predictions:
        kl_values_batch = []

        for i in range(pred.shape[0]):
            kl_values, mean_kl = kl_divergence(pred[i], ensemble_output[i])
            kl_values_batch.append(mean_kl)
            # Use to show scatter plot
            # kl_values_batch.append(kl_values)

        # TODO kl_values batch contains the mean kl for each sample
        students_kl.append(temperature**2 * kl_values_batch)
        '''
        # Create a scatter plot for each array
        for i, values in enumerate(kl_values_batch):
            # New plot for each kl_set
            plt.figure(i)

            plt.scatter(range(len(values)), values, label=f'Sample {i + 1}')

            plt.title(f'Scatter plot for sample {i + 1}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig("kl_div_"+str(i)+".jpg")                
            plt.close()
        '''
    return students_kl


def multiloss_function(peer_networks_n, ensemble_output, x_train, student_predictions, temperature, student_losses, distillation_strength):

    # Ensemble CTC loss
    ensemble_loss = compute_ensemble_loss(ensemble_output, x_train)
    print("Ensemble mean loss: {}".format(ensemble_loss))

    students_kl = kd_loss(student_predictions, ensemble_output, temperature)

    student_losses_sum = 0
    for s in range(peer_networks_n):

        # TODO check mean of students kl
        res = student_losses[s] + (distillation_strength * np.mean(students_kl[s]))
        student_losses_sum += res

    multiloss_value = ensemble_loss + student_losses_sum

    return multiloss_value