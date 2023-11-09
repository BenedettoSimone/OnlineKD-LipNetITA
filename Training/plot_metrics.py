import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_losses(run_name):

    loss_file = pd.read_csv(f'logs/training-{run_name[0]}.csv')

    train_loss = loss_file['ctc_loss']
    val_loss = loss_file['val_ctc_loss']

    plt.plot(train_loss,  color='red')
    plt.plot(val_loss,  color='blue')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name[0]}/losses.jpg")
    plt.close()


def plot_val_metrics(run_name):
    metrics_file = pd.read_csv(f'results/{run_name[0]}/stats.csv')

    wer = metrics_file['Mean WER (Norm)']
    cer = metrics_file['Mean CER (Norm)']

    plt.plot(wer, color='green')
    plt.plot(cer, color='orange')

    min_wer_index = wer.idxmin()
    min_cer_index = cer.idxmin()
    min_wer_value = wer[min_wer_index]
    min_cer_value = cer[min_cer_index]


    legend1 = plt.legend(
        [f'Min WER: {min_wer_value:.2f} (Epoch {min_wer_index})',
         f'Min CER: {min_cer_value:.2f} (Epoch {min_cer_index})'],
        loc='upper right'
    )



    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['WER', 'CER'], loc='upper left')
    #plt.show()
    plt.gca().add_artist(legend1)
    plt.savefig(f"plots/{run_name[0]}/val_metrics.jpg")
    plt.close()


def plot_losses_kd(run_name):

    loss_file = pd.read_csv(f'logs/training-{run_name[0]}.csv')

    train_loss_0 = loss_file['train_loss_s0']
    val_loss_0 = loss_file['val_loss_s0']
    train_loss_1 = loss_file['train_loss_s1']
    val_loss_1 = loss_file['val_loss_s1']

    plt.plot(train_loss_0,  color='red')
    plt.plot(val_loss_0,  color='blue')
    plt.plot(train_loss_1, color='green')
    plt.plot(val_loss_1, color='orange')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['train_loss_s0', 'val_loss_s0','train_loss_s1', 'val_loss_s1'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name[0]}/losses.jpg")
    plt.close()


def plot_val_metrics_kd(run_name):

    metrics_file = pd.read_csv(f'results/{run_name[0]}/s0/stats.csv')
    metrics_file1 = pd.read_csv(f'results/{run_name[0]}/s1/stats.csv')

    wer = metrics_file['Mean WER (Norm)']
    cer = metrics_file['Mean CER (Norm)']
    wer1 = metrics_file1['Mean WER (Norm)']
    cer1 = metrics_file1['Mean CER (Norm)']

    plt.plot(wer, color='green')
    plt.plot(cer, color='orange')
    plt.plot(wer1, color='blue')
    plt.plot(cer1, color='red')

    min_wer_index = wer.idxmin()
    min_cer_index = cer.idxmin()
    min_wer1_index = wer1.idxmin()
    min_cer1_index = cer1.idxmin()

    min_wer_value = wer[min_wer_index]
    min_cer_value = cer[min_cer_index]
    min_wer1_value = wer1[min_wer1_index]
    min_cer1_value = cer1[min_cer1_index]

    legend1 = plt.legend(
        [f'Min WER: {min_wer_value:.2f} (Epoch {min_wer_index})',
         f'Min CER: {min_cer_value:.2f} (Epoch {min_cer_index})',
         f'Min WER1: {min_wer1_value:.2f} (Epoch {min_wer1_index})',
         f'Min CER1: {min_cer1_value:.2f} (Epoch {min_cer1_index})'],
        loc='upper right'
    )

    #plt.grid(True)

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    legend2 = plt.legend(['WER', 'CER', 'WER1', 'CER1'], loc='upper left')

    plt.gca().add_artist(legend1)

    plt.savefig(f"plots/{run_name[0]}/val_metrics.jpg")
    plt.close()


def plot_train_metrics_kd(run_name):
    metrics_file = pd.read_csv(f'results/{run_name[0]}/training_metrics.csv')

    l0 = metrics_file['Loss student 0']
    l1 = metrics_file['Loss student 1']
    el = metrics_file['Ensemble loss']
    ml = metrics_file['Multiloss value']

    plt.plot(l0, color='green')
    plt.plot(l1, color='orange')
    plt.plot(el, color='blue')
    plt.plot(ml, color='red')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Loss 0', 'Loss 1', 'Ensemble loss', 'Multiloss'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name[0]}/training_metrics.jpg")
    plt.close()


if __name__ == '__main__':
    # Vanilla Training 32bs (isKD)
    #run_name = ("2023_10_26_09_22_27", False)

    # Vanilla Training-128 32bs
    #run_name = ("2023_10_28_11_30_29", False)

    # KD Training 16bs
    #run_name = ("2023_10_31_18_23_34", True)

    # Vanilla Training-256 32bs
    #run_name = ("2023_11_02_10_55_00", False)

    # Vanilla Training 16bs
    run_name = ("2023_11_04_16_41_41", False)

    # Vanilla Training-256 16bs
    #run_name = ("2023_11_06_17_35_37", False)

    # Vanilla Training-128 16bs
    #run_name = ("2023_11_06_17_36_26", False)


    if not os.path.exists("plots"):
        os.mkdir("plots")

    run_folder = os.path.join("plots", run_name[0])
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)


    if run_name[1] is False:
        plot_losses(run_name)
        plot_val_metrics(run_name)

    else:
        plot_losses_kd(run_name)
        plot_val_metrics_kd(run_name)
        plot_train_metrics_kd(run_name)




