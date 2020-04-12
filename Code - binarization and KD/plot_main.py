import seaborn as sns
import torch

def remove_tensor(array):
    for i, value in enumerate(array):
        if type(value) == torch.Tensor:
            array[i] = value.item()

def max_values():
    a = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_a_double_shortcut_with_relu_finetuning_Xnor++.pt')
    max_a = max(a)
    print('method a: ' + str(max_a))


    b0 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0_scaling_kd_0.95.pt')
    b02 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.2_scaling_kd_0.95.pt')
    b04 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.4_scaling_kd_0.95.pt')
    b06 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.6_scaling_kd_0.95.pt')
    b08 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_0.8_scaling_kd_0.95.pt')
    b1 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_double_shortcut_with_relu_Xnor++scaling_tot_1_scaling_kd_0.95.pt')

    b0f = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0_Xnor++.pt')
    b02f = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.2_Xnor++.pt')
    b04f = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.4_Xnor++.pt')
    b06f = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.6_Xnor++.pt')
    b08f = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_0.8_Xnor++.pt')
    b1f = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_b_finetuning_double_shortcut_with_relu_tot_scaling_1_Xnor++.pt')


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
    print('maxb08f: ' + str(max(b08f)))
    print('maxb1f: ' + str(max(b1f)))

    c0 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.pt')
    c02 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.2.pt')
    c04 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.4.pt')
    c06 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.6.pt')
    c08 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_0.8.pt')
    c1 = torch.load('Results/all_methods_double_shortcut/validation_accuracy_method_c_double_shortcut_with_relu_Xnor++_lambda_1.pt')

    print('maxc0: ' + str(max(c0)))
    print('maxc02: ' + str(max(c02)))
    print('maxc04: ' + str(max(c04)))
    print('maxc06: ' + str(max(c06)))
    print('maxc08: ' + str(max(c08)))
    print('maxc1: ' + str(max(c1)))

    noMethod = torch.load('Results/all_methods_double_shortcut/validation_accuracy_no_method_double_shortcut_with_relu_Xnor++.pt')

    print('no method: ' + str(max(noMethod)))

def main():

    max_values()






if __name__ == '__main__':
    main()
