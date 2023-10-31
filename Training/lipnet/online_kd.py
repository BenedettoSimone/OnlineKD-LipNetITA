import tensorflow as tf
from shared_mlp import SharedMLP
import numpy as np
from layers import ctc_lambda_func
import matplotlib.pyplot as plt


def extract_features(statistics, logits, x_train, f1_array, f2_array, f3_array):


    y_pred = tf.nn.softmax(logits)

    # Extract features
    f1, f2, f3 = statistics.on_batch_end(y_pred, x_train)

    # concatenate feature of each student for each sample
    if f1_array.size > 0:
        f1_array = np.vstack(
            (f1_array, f1))  # (b_size, n_students) es. [[42.  9.], [16.  6.],...., [15.  5.]]
    else:
        f1_array = np.append(f1_array, np.array(f1))

    if f2_array.size > 0:
        f2_array = np.vstack((f2_array, f2))
    else:
        f2_array = np.append(f2_array, np.array(f2))

    if f3_array.size > 0:
        f3_array = np.vstack((f3_array, f3))
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
        # These weights are determined through an element-wise summation operation on the
        # outputs of the perceptron, followed by passing them through a sigmoid activation function
        sum_predictions_student_i = tf.reduce_sum(predictions_student_i, axis=1)
        weights_student_i = tf.math.sigmoid(sum_predictions_student_i)
        student_weights.append(weights_student_i)

    return student_weights


def compute_ensemble_output(student_logits, student_weights):
    ensemble_output = []

    # Get how many samples
    for i in range(student_logits[0][1].shape[0]):
        # Get weighted logits of sample
        weighted_logits = []
        weights_log_i = []

        # Get i logits of each student
        for student_idx, logits in student_logits:
            # Multiply each tensor for the student weight for the corresponding logits
            weighted_logits.append(logits[i] * student_weights[student_idx][i])
            weights_log_i.append(student_weights[student_idx][i])

        # For each sample (i.e. pred1 of s1, pred1 of s2, pred1 of s3)
        # (pred * w1 + pred * w2 + pred * w3) / w1+w2+w3
        # Sum each student tensor
        ensemble_logits = np.sum(weighted_logits, axis=0)
        ensemble_logits = ensemble_logits / np.sum(weights_log_i)
        ensemble_output.append(ensemble_logits)

    ensemble_output = np.array(ensemble_output)

    return ensemble_output


def compute_ensemble_loss(ensemble_output, x_train):

    labels = tf.convert_to_tensor(x_train['the_labels'])
    label_length = tf.convert_to_tensor(x_train['label_length'].reshape(-1, 1))
    input_length = tf.convert_to_tensor(x_train['input_length'].reshape(-1, 1))
    ensemble_output = tf.convert_to_tensor(ensemble_output)

    ensemble_output = tf.nn.softmax(ensemble_output)
    ensemble_ctc_loss = ctc_lambda_func([ensemble_output, labels, input_length, label_length])

    # Compute the mean CTC loss
    ensemble_loss = tf.reduce_mean(ensemble_ctc_loss, axis=0)

    return ensemble_loss

def kl_divergence(ensemble_output, student_prediction):
    sequence_kl_divergence = []

    for i in range(student_prediction.shape[0]):
        p = ensemble_output[i]
        q = student_prediction[i]

        kl_value = np.sum(p * np.log(p / q))
        sequence_kl_divergence.append(kl_value)

    # compute sum KL div over sequence
    sum_kl_divergence = np.sum(sequence_kl_divergence)
    return sequence_kl_divergence, sum_kl_divergence


def kd_loss(student_logits, ensemble_output, temperature):

    students_kl = []

    # Get i logits of each student and compute kl divergence
    for student_idx, logits in student_logits:
        kl_values_batch = []

        for i in range(logits.shape[0]):

            logits_i =  tf.nn.softmax(tf.math.divide(logits[i], temperature))
            ensemble_output_i = tf.nn.softmax(tf.math.divide(ensemble_output[i], temperature))

            kl_values, sum_kl = kl_divergence(ensemble_output_i, logits_i)
            kl_values_batch.append(sum_kl)
            # Use to show scatter plot
            # kl_values_batch.append(kl_values)

        # kl_values batch contains the kl value for each sample
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


def multiloss_function(peer_networks_n, ensemble_output, x_train, student_logits, temperature, student_losses, distillation_strength):

    # Ensemble CTC loss
    ensemble_loss = compute_ensemble_loss(ensemble_output, x_train)
    print("Ensemble mean loss: {}".format(ensemble_loss))

    students_kl = kd_loss(student_logits, ensemble_output, temperature)

    student_losses_sum = 0
    for s in range(peer_networks_n):

        # use mean of the kl values over samples
        res = student_losses[s] + (distillation_strength * np.mean(students_kl[s]))
        student_losses_sum += res

    multiloss_value = ensemble_loss + student_losses_sum

    return ensemble_loss, multiloss_value