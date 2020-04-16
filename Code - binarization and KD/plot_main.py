import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np


def remove_tensor(array):
    for i, value in enumerate(array):
        if type(value) == torch.Tensor:
            array[i] = value.item()


def binary_xnor_xnorpp():
    binary_a_train_loss = torch.load('Results/train_loss_method_a_double_shortcut_with_relu_binary_20200413.pt')
    binary_a_valid_loss = torch.load('Results/validation_loss_method_a_double_shortcut_with_relu_binary_20200413.pt')
    binary_a_train_acc = torch.load('Results/train_accuracy_method_a_double_shortcut_with_relu_binary_20200413.pt')
    binary_a_valid_acc = torch.load('Results/validation_accuracy_method_a_double_shortcut_with_relu_binary_20200413.pt')
    binary_no_train_acc = torch.load('Results/train_accuracy_no_method_double_shortcut_with_relu_binary.pt')
    binary_no_valid_acc = torch.load('Results/validation_accuracy_no_method_double_shortcut_with_relu_binary.pt')
    binary_no_train_loss = torch.load('Results/train_loss_no_method_double_shortcut_with_relu_binary.pt')
    binary_no_valid_loss = torch.load('Results/validation_loss_no_method_double_shortcut_with_relu_binary.pt')

    xnor_a_train_loss = torch.load('Results/train_loss_method_a_double_shortcut_with_relu_Xnor_20200414.pt')
    xnor_a_valid_loss = torch.load('Results/validation_loss_method_a_double_shortcut_with_relu_Xnor_20200414.pt')
    xnor_a_train_acc = torch.load('Results/train_accuracy_method_a_double_shortcut_with_relu_Xnor_20200414.pt')
    xnor_a_valid_acc = torch.load('Results/validation_accuracy_method_a_double_shortcut_with_relu_Xnor_20200414.pt')
    xnor_no_train_acc = torch.load('Results/train_accuracy_no_method_double_shortcut_with_relu_Xnor.pt')
    xnor_no_valid_acc = torch.load('Results/validation_accuracy_no_method_double_shortcut_with_relu_Xnor.pt')
    xnor_no_train_loss = torch.load('Results/train_loss_no_method_double_shortcut_with_relu_Xnor.pt')
    xnor_no_valid_loss = torch.load('Results/validation_loss_no_method_double_shortcut_with_relu_Xnor.pt')

    xnorpp_a_train_loss = torch.load('Results/train_loss_method_a_double_shortcut_with_relu_Xnor++_20200414.pt')
    xnorpp_a_valid_loss = torch.load('Results/validation_loss_method_a_double_shortcut_with_relu_Xnor++_20200414.pt')
    xnorpp_a_train_acc = torch.load('Results/train_accuracy_method_a_double_shortcut_with_relu_Xnor++_20200414.pt')
    xnorpp_a_valid_acc = torch.load('Results/validation_accuracy_method_a_double_shortcut_with_relu_Xnor++_20200414.pt')
    xnorpp_no_train_acc = torch.load('Results/train_accuracy_no_method_double_shortcut_with_relu_Xnor++.pt')
    xnorpp_no_valid_acc = torch.load('Results/validation_accuracy_no_method_double_shortcut_with_relu_Xnor++.pt')
    xnorpp_no_train_loss = torch.load('Results/train_loss_no_method_double_shortcut_with_relu_Xnor++.pt')
    xnorpp_no_valid_loss = torch.load('Results/validation_loss_no_method_double_shortcut_with_relu_Xnor++.pt')

    xnorppF_a_train_loss = torch.load(
        'Results/train_loss_method_a_double_shortcut_with_relu_factorized_Xnor++_20200413.pt')
    xnorppF_a_valid_loss = torch.load(
        'Results/validation_loss_method_a_double_shortcut_with_relu_factorized_Xnor++_20200413.pt')
    xnorppF_a_train_acc = torch.load(
        'Results/train_accuracy_method_a_double_shortcut_with_relu_factorized_Xnor++_20200413.pt')
    xnorppF_a_valid_acc = torch.load(
        'Results/validation_accuracy_method_a_double_shortcut_with_relu_factorized_Xnor++_20200413.pt')
    xnorppF_no_train_acc = torch.load('Results/train_accuracy_no_method_double_shortcut_with_relu_factorized_Xnor++.pt')
    xnorppF_no_valid_acc = torch.load(
        'Results/validation_accuracy_no_method_double_shortcut_with_relu_factorized_Xnor++.pt')
    xnorppF_no_train_loss = torch.load('Results/train_loss_no_method_double_shortcut_with_relu_factorized_Xnor++.pt')
    xnorppF_no_valid_loss = torch.load(
        'Results/validation_loss_no_method_double_shortcut_with_relu_factorized_Xnor++.pt')

    max_epochs_a = 90
    max_epoch_finetuning = 110

    epochs_training = np.arange(max_epochs_a)
    epochs_finetuning = np.arange(max_epoch_finetuning)

    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("paper")

    with sns.color_palette("bright"):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax0 = ax[0]
        ax1 = ax[1]
        ax2 = ax[2]

        ax0.plot(epochs_training, binary_a_train_loss[:max_epochs_a], 'C1', linestyle='dotted')
        ax0.plot(epochs_training, binary_a_valid_loss[:max_epochs_a], 'C1', linestyle='solid', label='Binary')
        ax0.plot(epochs_training, xnor_a_train_loss[:max_epochs_a], 'C2', linestyle='dotted', )
        ax0.plot(epochs_training, xnor_a_valid_loss[:max_epochs_a], 'C2', linestyle='solid', label='Xnor')
        ax0.plot(epochs_training, xnorpp_a_train_loss[:max_epochs_a], 'C3', linestyle='dotted')
        ax0.plot(epochs_training, xnorpp_a_valid_loss[:max_epochs_a], 'C3', linestyle='solid', label='Xnor++, Γ')
        ax0.plot(epochs_training, xnorppF_a_train_loss[:max_epochs_a], 'C0', linestyle='dotted')
        ax0.plot(epochs_training, xnorppF_a_valid_loss[:max_epochs_a], 'C0', linestyle='solid', label='Xnor++, αβɣ')

        ax0.legend()

        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('MSE loss')
        ax0.set_title('Training, method a)')

        ax1.plot(epochs_finetuning, binary_a_train_loss[max_epochs_a:], 'C1', linestyle='dotted')
        ax1.plot(epochs_finetuning, binary_a_valid_loss[max_epochs_a:], 'C1', linestyle='solid', label='Binary')
        ax1.plot(epochs_finetuning, xnor_a_train_loss[max_epochs_a:], 'C2', linestyle='dotted', )
        ax1.plot(epochs_finetuning, xnor_a_valid_loss[max_epochs_a:], 'C2', linestyle='solid', label='Xnor')
        ax1.plot(epochs_finetuning, xnorpp_a_train_loss[max_epochs_a:], 'C3', linestyle='dotted')
        ax1.plot(epochs_finetuning, xnorpp_a_valid_loss[max_epochs_a:], 'C3', linestyle='solid', label='Xnor++, Γ')
        ax1.plot(epochs_finetuning, xnorppF_a_train_loss[max_epochs_a:], 'C0', linestyle='dotted')
        ax1.plot(epochs_finetuning, xnorppF_a_valid_loss[max_epochs_a:], 'C0', linestyle='solid', label='Xnor++, αβɣ')

        ax1.legend()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross entropy loss')
        ax1.set_title('Regular training after method a) - loss')

        ax2.plot(epochs_finetuning, binary_a_train_acc, 'C1', linestyle='dotted')
        ax2.plot(epochs_finetuning, binary_a_valid_acc, 'C1', linestyle='solid', label='Binary')
        ax2.plot(epochs_finetuning, xnor_a_train_acc, 'C2', linestyle='dotted', )
        ax2.plot(epochs_finetuning, xnor_a_valid_acc, 'C2', linestyle='solid', label='Xnor')
        ax2.plot(epochs_finetuning, xnorpp_a_train_acc, 'C3', linestyle='dotted')
        ax2.plot(epochs_finetuning, xnorpp_a_valid_acc, 'C3', linestyle='solid', label='Xnor++, Γ')
        ax2.plot(epochs_finetuning, xnorppF_a_train_acc, 'C0', linestyle='dotted')
        ax2.plot(epochs_finetuning, xnorppF_a_valid_acc, 'C0', linestyle='solid', label='Xnor++, αβɣ')

        ax2.legend()

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Regular training after method a) - accuracy')

        plt.tight_layout(h_pad=1)
        fig.savefig('bin_xnor_xnorpp_method_a.eps', format='eps')

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax0 = ax[0]
        ax1 = ax[1]

        ax0.plot(epochs_finetuning, binary_no_train_loss, 'C1', linestyle='dotted')
        ax0.plot(epochs_finetuning, binary_no_valid_loss, 'C1', linestyle='solid', label='Binary')
        ax0.plot(epochs_finetuning, xnor_no_train_loss, 'C2', linestyle='dotted', )
        ax0.plot(epochs_finetuning, xnor_no_valid_loss, 'C2', linestyle='solid', label='Xnor')
        ax0.plot(epochs_finetuning, xnorpp_no_train_loss, 'C3', linestyle='dotted')
        ax0.plot(epochs_finetuning, xnorpp_no_valid_loss, 'C3', linestyle='solid', label='Xnor++, Γ')
        ax0.plot(epochs_finetuning, xnorppF_no_train_loss, 'C0', linestyle='dotted')
        ax0.plot(epochs_finetuning, xnorppF_no_valid_loss, 'C0', linestyle='solid', label='Xnor++, αβɣ')

        ax0.legend()

        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('Cross entropy loss')
        ax0.set_title('Regular training - loss')

        ax1.plot(epochs_finetuning, binary_no_train_acc, 'C1', linestyle='dotted')
        ax1.plot(epochs_finetuning, binary_no_valid_acc, 'C1', linestyle='solid', label='Binary')
        ax1.plot(epochs_finetuning, xnor_no_train_acc, 'C2', linestyle='dotted', )
        ax1.plot(epochs_finetuning, xnor_no_valid_acc, 'C2', linestyle='solid', label='Xnor')
        ax1.plot(epochs_finetuning, xnorpp_no_train_acc, 'C3', linestyle='dotted')
        ax1.plot(epochs_finetuning, xnorpp_no_valid_acc, 'C3', linestyle='solid', label='Xnor++, Γ')
        ax1.plot(epochs_finetuning, xnorppF_no_train_acc, 'C0', linestyle='dotted')
        ax1.plot(epochs_finetuning, xnorppF_no_valid_acc, 'C0', linestyle='solid', label='Xnor++, αβɣ')

        ax1.legend()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross entropy loss')
        ax1.set_title('Regular training - accuracy')

        plt.tight_layout(h_pad=2)
        fig.savefig('bin_xnor_xnorpp_no_method.eps', format='eps')

        plt.show()


def max_values_a_b_c():
    a = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_a_double_shortcut_with_relu_finetuning_Xnor++.pt')
    max_a = max(a)
    print('method a: ' + str(max_a))

    valid_acc = torch.load('Results/validation_accuracy_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')
    max_a = max(valid_acc)
    print('method a: ' + str(max_a))

    results_a = np.array([max_a, max_a])
    param_a = [0, 1]

    b0 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0_scaling_kd_0.95.pt')
    b02 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.2_scaling_kd_0.95.pt')
    b04 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.4_scaling_kd_0.95.pt')
    b06 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.6_scaling_kd_0.95.pt')
    b08 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.8_scaling_kd_0.95.pt')
    b1 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_1_scaling_kd_0.95.pt')

    b0f = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0_Xnor++.pt')
    b02f = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.2_Xnor++.pt')
    b04f = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.4_Xnor++.pt')
    b06f = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.6_Xnor++.pt')
    b075 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.75_scaling_kd_0.95.pt')
    b08f = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.8_Xnor++.pt')
    b1f = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_1_Xnor++.pt')

    # print('maxb0:' + str(max(b0)))
    # print('maxb02:' + str(max(b02)))
    # print('maxb04:' + str(max(b04)))
    # print('maxb06:' + str(max(b06)))
    # print('maxb08:' + str(max(b08)))
    # print('maxb1:' + str(max(b1)))

    print('maxb0f: ' + str(max(b0f)))
    print('maxb02f: ' + str(max(b02f)))
    print('maxb04f: ' + str(max(b04f)))
    print('maxb06f: ' + str(max(b06f)))
    print('maxb075:' + str(max(b075)))
    print('maxb08f: ' + str(max(b08f)))
    print('maxb1f: ' + str(max(b1f)))

    results_b = np.array([max(b0f), max(b02f), max(b04f), max(b06f), max(b08f), max(b1f)])
    param_b = [0, 0.2, 0.4, 0.6, 0.8, 1]

    c0 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.pt')
    c02 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.2.pt')
    c04 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.4.pt')
    c05 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++.pt')
    c06 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.6.pt')
    c08 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.8.pt')
    c1 = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_1.pt')

    print('maxc0: ' + str(max(c0)))
    print('maxc02: ' + str(max(c02)))
    print('maxc04: ' + str(max(c04)))
    print('maxc05: ' + str(max(c05)))
    print('maxc06: ' + str(max(c06)))
    print('maxc08: ' + str(max(c08)))
    print('maxc1: ' + str(max(c1)))

    results_c = np.array([max(c0), max(c02), max(c04), max(c05), max(c06), max(c08), max(c1)])
    param_c = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]

    noMethod = torch.load(
        'Results/all_methods_double_shortcut/validation_accuracy_no_method_double_shortcut_with_relu_Xnor++.pt')

    print('no method: ' + str(max(noMethod)))

    results_no = [max(noMethod), max(noMethod)]
    param_no = [0, 1]

    sns.set()
    sns.set_style("white")
    sns.set_context("talk")
    with sns.color_palette("muted"):
        # sns.set_style("ticks")
        # sns.set_context("paper")

        fig, axs = plt.subplots(1,2, figsize=(10, 5))

        ax=axs[0]
        axs[1].set_visible(False)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        a = ax.plot(param_a, results_a, label='method a', linestyle='dashed', linewidth=3)
        b = ax.plot(param_b, results_b, label='method b', linestyle='dotted', linewidth=3.5)
        c = ax.plot(param_c, results_c, label='method c', linestyle='dashdot', linewidth=3)
        d = ax.plot(param_no, results_no, label='no method', linewidth=3, color='black')

        ax.set_title('Accuracy for method a), b) and c)')
        ax.set_xlabel('Scaling parameter')
        ax.set_ylabel('Accuracy, %')

        # Put a legend to the right of the current axis
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        fig.tight_layout()

        fig.savefig('accuracy_a_b_c.eps', format='eps')

        # plt.tight_layout(h_pad=10)
        plt.show()


def method_a():
    a_acc = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_a_double_shortcut_with_relu_finetuning_Xnor++.pt')
    max_a = max(a_acc)
    print('method a: ' + str(max_a))

    train_acc = torch.load('Results/train_accuracy_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')
    valid_acc = torch.load('Results/validation_accuracy_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')
    train_loss = torch.load('Results/train_loss_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')
    valid_loss = torch.load('Results/validation_loss_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')

    max_epoch_training = 90
    max_epoch_finetuning = 60
    epochs_training = np.arange(max_epoch_training)
    epochs_finetuning = np.arange(max_epoch_finetuning)

    sns.set()
    sns.set_style("whitegrid")
    #sns.set_context("talk")
    with sns.color_palette("bright"):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.15))
        ax0 = axs[0]
        ax1 = axs[1]
        ax2 = axs[2]

        ax0.plot(epochs_training, train_loss[:max_epoch_training], linestyle='dashed', label='train')
        ax0.plot(epochs_training, valid_loss[:max_epoch_training], linestyle='solid', label='validation')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('MSE loss')
        ax0.set_title('Training using method a)')
        ax0.legend(loc='upper left')

        ax1.plot(epochs_finetuning, train_loss[max_epoch_training:], linestyle='dashed', label='train')
        ax1.plot(epochs_finetuning, valid_loss[max_epoch_training:], linestyle='solid', label='validation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross entropy loss')
        ax1.set_title('Training after method a) - loss')
        ax1.legend(loc='upper right')

        ax2.plot(epochs_finetuning, train_acc, linestyle='dashed', label='train')
        ax2.plot(epochs_finetuning, valid_acc, linestyle='solid', label='validation')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Cross entropy loss')
        ax2.set_title('Training after method a) - accuracy')
        ax2.legend(loc='lower right')

        plt.tight_layout()
        fig.savefig('method_a_training.eps', format='eps')

        plt.show()


def method_b():
    a_acc = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_a_double_shortcut_with_relu_finetuning_Xnor++.pt')
    max_a = max(a_acc)
    print('method a: ' + str(max_a))

    train_acc = torch.load('Results/all_methods_double_shortcut/train_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.6_scaling_kd_0.95.pt')
    valid_acc = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.6_scaling_kd_0.95.pt')
    train_loss = torch.load('Results/all_methods_double_shortcut/train_loss_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.6_scaling_kd_0.95.pt')
    valid_loss = torch.load('Results/all_methods_double_shortcut/validation_loss_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.6_scaling_kd_0.95.pt')

    train_finetuning_acc = torch.load('Results/all_methods_double_shortcut/train_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.6_Xnor++.pt')
    validation_finetuning_acc = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.6_Xnor++.pt')
    train_finetuning_loss = torch.load('Results/all_methods_double_shortcut/train_loss_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.6_Xnor++.pt')
    validation_finetuning_loss = torch.load('Results/all_methods_double_shortcut/validation_loss_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.6_Xnor++.pt')

    max_epoch_training = len(train_loss)
    max_epoch_finetuning = 60
    epochs_training = np.arange(max_epoch_training)
    epochs_finetuning = np.arange(max_epoch_finetuning)

    sns.set()
    sns.set_style("whitegrid")
    #sns.set_context("talk")
    with sns.color_palette("bright"):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.15))
        ax0 = axs[0]
        ax1 = axs[1]
        ax2 = axs[2]

        ax0.plot(epochs_training, train_loss, linestyle='dashed', label='train')
        ax0.plot(epochs_training, valid_loss, linestyle='solid', label='validation')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('MSE loss')
        ax0.set_title('Training using method b)')
        ax0.legend(loc='upper right')

        ax1.plot(epochs_finetuning, train_finetuning_loss, linestyle='dashed', label='train')
        ax1.plot(epochs_finetuning, validation_finetuning_loss, linestyle='solid', label='validation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross entropy loss')
        ax1.set_title('Training after method b) - loss')
        ax1.legend(loc='upper right')

        ax2.plot(epochs_finetuning, train_finetuning_acc, linestyle='dashed', label='train')
        ax2.plot(epochs_finetuning, validation_finetuning_acc, linestyle='solid', label='validation')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Cross entropy loss')
        ax2.set_title('Training after method b) - accuracy')
        ax2.legend(loc='lower right')

        plt.tight_layout()
        fig.savefig('method_b_training.eps', format='eps')

        plt.show()


def method_c():
    a_acc = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_a_double_shortcut_with_relu_finetuning_Xnor++.pt')
    max_a = max(a_acc)
    print('method a: ' + str(max_a))

    train_acc = torch.load('Results/all_methods_double_shortcut/train_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.4.pt')
    valid_acc = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.4.pt')
    train_loss = torch.load('Results/all_methods_double_shortcut/train_loss_method_c_double_shortcut_with_relu_Xnor++_lambda_0.4.pt')
    valid_loss = torch.load('Results/all_methods_double_shortcut/validation_loss_method_c_double_shortcut_with_relu_Xnor++_lambda_0.4.pt')

    max_epoch_training = 120
    max_epoch_finetuning = 60
    epochs_training = np.arange(max_epoch_training)
    epochs_finetuning = np.arange(max_epoch_finetuning)

    sns.set()
    sns.set_style("whitegrid")
    #sns.set_context("talk")
    with sns.color_palette("bright"):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.15))
        ax0 = axs[0]
        ax1 = axs[1]
        ax2 = axs[2]

        ax0.plot(epochs_training, train_loss[:max_epoch_training], linestyle='dashed', label='train')
        ax0.plot(epochs_training, valid_loss[:max_epoch_training], linestyle='solid', label='validation')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('MSE loss')
        ax0.set_title('Training using method c)')
        ax0.legend(loc='upper right')

        ax1.plot(epochs_finetuning, train_loss[max_epoch_training:], linestyle='dashed', label='train')
        ax1.plot(epochs_finetuning, valid_loss[max_epoch_training:], linestyle='solid', label='validation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross entropy loss')
        ax1.set_title('Training after method c) - loss')
        ax1.legend(loc='upper right')

        ax2.plot(epochs_finetuning, train_acc[max_epoch_training:], linestyle='dashed', label='train')
        ax2.plot(epochs_finetuning, valid_acc[max_epoch_training:], linestyle='solid', label='validation')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Cross entropy loss')
        ax2.set_title('Training after method c) - accuracy')
        ax2.legend(loc='lower right')

        plt.tight_layout()
        fig.savefig('method_c_training.eps', format='eps')

        plt.show()


def plot_traing_a(train_loss, valid_loss, train_acc, valid_acc, filename, suptitle):
    max_epoch_training = 90
    max_epoch_finetuning = 60
    epochs_training = np.arange(max_epoch_training)
    epochs_finetuning = np.arange(max_epoch_finetuning)

    sns.set()
    sns.set_style("whitegrid")
    # sns.set_context("talk")
    with sns.color_palette("bright"):
        fig, axs = plt.subplots(2, 3, figsize=(9, 5.5))
        ax0 = axs[1, 0]
        ax1 = axs[1, 1]
        ax2 = axs[1, 2]

        axs[0, 0].set_visible(False)
        axs[0, 1].set_visible(False)
        axs[0, 2].set_visible(False)


        ax0.plot(epochs_training, train_loss[:max_epoch_training], linestyle='dashed', label='train')
        ax0.plot(epochs_training, valid_loss[:max_epoch_training], linestyle='solid', label='validation')
        ax0.set_xlabel('Epochs')
        ax0.set_ylabel('MSE loss')
        ax0.set_title('Training using method a)')
        ax0.legend(loc='upper left')

        ax1.plot(epochs_finetuning, train_loss[max_epoch_training:], linestyle='dashed', label='train')
        ax1.plot(epochs_finetuning, valid_loss[max_epoch_training:], linestyle='solid', label='validation')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross entropy loss')
        ax1.set_title('Training after method a) - loss')
        ax1.legend(loc='upper right')

        ax2.plot(epochs_finetuning, train_acc, linestyle='dashed', label='train')
        ax2.plot(epochs_finetuning, valid_acc, linestyle='solid', label='validation')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Cross entropy loss')
        ax2.set_title('Training after method a) - accuracy')
        ax2.legend(loc='lower right')

        plt.suptitle(suptitle, fontsize=14, y=0.6)

        plt.tight_layout()
        fig.savefig(filename + '.eps', format='eps')

        plt.show()


def single_double_relu():

    naive_train_loss = torch.load('Results/train_loss_method_a_naive_block_Xnor++_20200406.pt')
    naive_valid_loss = torch.load('Results/validation_loss_method_a_naive_block_Xnor++_20200406.pt')
    naive_train_acc = torch.load('Results/train_accuracy_method_a_naive_block_Xnor++_20200406.pt')
    naive_valid_acc = torch.load('Results/validation_accuracy_method_a_naive_block_Xnor++_20200406.pt')

    relu_single_train_loss = torch.load('Results/train_loss_method_a_with_relu_blockXnor++_20200407.pt')
    relu_single_valid_loss = torch.load('Results/validation_loss_method_a_with_relu_blockXnor++_20200407.pt')
    relu_single_train_acc = torch.load('Results/train_accuracy_method_a_with_relu_blockXnor++_20200407.pt')
    relu_single_valid_acc = torch.load('Results/validation_accuracy_method_a_with_relu_blockXnor++_20200407.pt')

    abs_single_train_loss = torch.load('Results/train_loss_method_a_with_abs_blockXnor++_20200407.pt')
    abs_single_valid_loss = torch.load('Results/validation_loss_method_a_with_abs_blockXnor++_20200407.pt')
    abs_single_train_acc = torch.load('Results/train_accuracy_method_a_with_abs_blockXnor++_20200407.pt')
    abs_single_valid_acc = torch.load('Results/validation_accuracy_method_a_with_abs_blockXnor++_20200407.pt')

    relu_double_train_acc = torch.load('Results/train_accuracy_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')
    relu_double_valid_acc = torch.load('Results/validation_accuracy_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')
    relu_double_train_loss = torch.load('Results/train_loss_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')
    relu_double_valid_loss = torch.load('Results/validation_loss_method_a_double_shortcut_with_relu_Xnor++_20200410.pt')

    plot_traing_a(naive_train_loss, naive_valid_loss, naive_train_acc, naive_valid_acc, 'naive_method_a', 'Training with method a) using naive ResNet block')
    plot_traing_a(relu_single_train_loss, relu_single_valid_loss, relu_single_train_acc, relu_single_valid_acc, 'relu_one_shortcut_a', 'Training with method a) using single ReLU shortcut')
    plot_traing_a(abs_single_train_loss, abs_single_valid_loss, abs_single_train_acc, abs_single_valid_acc, 'abs_one_shortcut_a', 'Training with method a) using absolute value instead of ReLU, single shortcut')
    plot_traing_a(relu_double_train_loss, relu_double_valid_loss, relu_double_train_acc, relu_double_valid_acc, 'relu_double_shortcut_a', 'Training with method a) using double ReLU shortcut')

def main():
    single_double_relu()


if __name__ == '__main__':
    main()
