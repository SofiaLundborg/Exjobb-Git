

def copy_parameters(net, original_net):
    ''' If net and original net have the same architecture, it copies all data in parameters in original net to net '''
    net_parameters = list(net.parameters())
    org_net_parameters = list(original_net.parameters())

    i = 0
    for net_parameter in net_parameters:
        if net_parameter.data.size() == org_net_parameters[i].size():    # There are no large gamma in fp_net
            net_parameter.data = org_net_parameters[i].clone()
            i += 1
    if not (i == len(org_net_parameters)):
        print('something wrong when copying the parameters')


def calculate_output_size(input_size, kernel_size, stride, padding):
    output_size = int((input_size - kernel_size + 2 * padding) / stride + 1)
    return output_size


