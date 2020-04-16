from collections import OrderedDict


def copy_parameters(net, original_net):
    ''' If net and original net have the same architecture, it copies all data in parameters in original net to net '''
    net_parameters = list(net.parameters())
    org_net_parameters = list(original_net.parameters())

    i = 0
    for net_parameter in net_parameters:
        if net_parameter.data.size() == org_net_parameters[i].size():    # There are no large gamma in fp_net
            net_parameter.data = org_net_parameters[i].clone().detach()
            #net_parameters[i] = net_parameter[i]
            #net_parameter = copy.deepcopy(org_net_parameters)
            i += 1
    if not (i == len(org_net_parameters)):
        print('something wrong when copying the parameters')


def calculate_output_size(input_size, kernel_size, stride, padding):
    output_size = int((input_size - kernel_size + 2 * padding) / stride + 1)
    return output_size


def change_loaded_checkpoint(checkpoint, student_net):
    student_dict = student_net.state_dict()

    new_checkpoint = OrderedDict()
    for key in checkpoint:
        str_key = key
        str_key = str_key.replace('conv1.', 'conv1.conv2d.')
        str_key = str_key.replace('conv2.', 'conv2.conv2d.')
        str_key = str_key.replace('downsample', 'shortcut')
        str_key = str_key.replace('fc', 'linear')

        new_checkpoint[str_key] = checkpoint[key]

    for key_student in student_dict:
        if key_student not in new_checkpoint:
            new_checkpoint[key_student] = student_dict[key_student]

    return new_checkpoint

