import torch
import torch.nn as nn
import torch.cuda
import torch.optim
import torch.utils.data


class Loss(nn.Module):
    def __init__(self, scaling_factor_total, scaling_factor_kd, temperature_kd):
        super(Loss, self).__init__()

        self.beta = scaling_factor_total
        self.kd_loss = KdLoss(temperature_kd, scale=True, weight=None, alpha=scaling_factor_kd, size_average=True)
        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.mse_loss = self.mse_loss.cuda()
            self.ce_loss = self.ce_loss.cuda()

    def forward(self, inputs, targets, student_net, teacher_net=None, intermediate_layers=None, lit_training=False, input_from_teacher=False, cut_network=None, training=True):

        if cut_network:
            output_student = student_net(inputs, intermediate_layers, cut_network)
            output_teacher = teacher_net(inputs, intermediate_layers, cut_network)

            return self.mse_loss(output_student, output_teacher)

        if lit_training:
            ir_loss = 0
            with torch.no_grad():
                teacher_net.eval()
                features_teacher, output_teacher = teacher_net(inputs, intermediate_layers)
                out_student = features_teacher[intermediate_layers[0]]
            for i_layer, layer in enumerate(['layer1', 'layer2', 'layer3']):
                section_student = getattr(student_net, layer)
                inp = features_teacher[intermediate_layers[i_layer]]
                inp = [inp, i_layer, None, None, None]
                if training:
                    if not input_from_teacher:
                        out_student = section_student(out_student)[0]
                    else:
                        out_student = section_student(inp)[0]
                else:
                    with torch.no_grad(): out_student = section_student(inp)[0]
                out_teacher = features_teacher[intermediate_layers[i_layer + 1]]
                ir_loss += self.mse_loss(out_student, out_teacher)

            output_student = student_net(inputs)
            loss_last_layer = self.kd_loss(output_student, output_teacher, targets)
            total_loss = self.beta * loss_last_layer + (1 - self.beta) * ir_loss

            return total_loss

        if intermediate_layers:
            features_student, output_student = student_net(inputs, intermediate_layers, cut_network)
        else:
            output_student = student_net(inputs, intermediate_layers, cut_network)
        if teacher_net:
            if intermediate_layers:
                teacher_net.eval()
                while torch.no_grad():
                    features_teacher, output_teacher = teacher_net(inputs, intermediate_layers, cut_network)
            else:
                output_teacher = teacher_net(inputs, intermediate_layers, cut_network)

            loss_intermediate_results = 0
            if intermediate_layers:
                for (key, value) in features_student.items():
                    loss_intermediate_results += self.mse_loss(features_student[key], features_teacher[key])

            loss_knowledge_distillation = self.kd_loss(output_student, output_teacher, targets)
            total_loss = self.beta*loss_knowledge_distillation + (1-self.beta)*loss_intermediate_results
        else:
            if training:
                total_loss = self.ce_loss(output_student, targets)
            else:
                with torch.no_grad():
                    total_loss = self.ce_loss(output_student, targets)
            if cut_network:
                print('Need teacher_network when cutting network')

        return total_loss


class KdLoss(nn.Module):
    def __init__(self, temperature=1.0, scale=True, weight=None, alpha=0.5,
                 size_average=True):
        super(KdLoss, self).__init__()
        self.temp = temperature
        self.scale = scale
        self.alpha = alpha
        self.size_average = size_average

        # Don't scale losses because they will be combined later
        self._hard_loss = nn.CrossEntropyLoss(weight=weight, size_average=False)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        if torch.cuda.is_available():
            self._hard_loss = self._hard_loss.cuda()
            self.softmax = self.softmax.cuda()
            self.log_softmax = self.log_softmax.cuda()

    def forward(self, inputs, soft_targets, hard_targets):
        # Figure soft target predictions
        _, preds = torch.max(soft_targets, dim=1, keepdim=False)

        # Calculate Cross Entropy with true targets
        hard_loss = self._hard_loss(inputs, hard_targets)

        # Calculate number of correct hard predictions
        hard = torch.nonzero(preds.ne(hard_targets).data)

        # Calculate Cross Entropy with soft targets
        hi_temp_inputs = self.log_softmax(inputs / self.temp)
        # Need high temperature probability distribution as target
        _soft_targets = self.softmax(soft_targets / self.temp)
        soft_cross_entropy = -(hi_temp_inputs * _soft_targets).sum(1)
        soft_loss = soft_cross_entropy.sum()

        unscaled_soft_loss = soft_loss.clone()
        # Scale to balance hard and soft loss functions
        if self.scale:
            soft_loss *= self.temp ** 2

        # Calculate soft targets Entropy
        soft_entropy = -1 * _soft_targets * torch.log(_soft_targets)
        soft_entropy[soft_entropy != soft_entropy] = 0
        soft_entropy = soft_entropy.sum(1)

        # Calculate Kullback-Leibler divergence
        soft_kl_divergence = soft_cross_entropy - soft_entropy

        # Calculate number of correct soft predictions
        soft = torch.nonzero(preds.eq(hard_targets).data)

        # Sum unscaled losses
        loss = sum([(1 - self.alpha) * soft_loss, self.alpha * hard_loss])
        if self.size_average:
            loss /= inputs.size(0)

        # loss.extra = OrderedDict()
        # loss.extra['alpha'] = self.alpha
        # loss.extra['nhard'] = len(hard)
        # loss.extra['nsoft'] = len(soft)
        # loss.extra['hard_loss'] = hard_loss.data[0]
        # loss.extra['soft_loss'] = soft_loss.data[0]
        # loss.extra['unscaled_soft_loss'] = unscaled_soft_loss.data[0]
        # loss.extra['soft_entropy_mean'] = soft_entropy.mean().data[0]
        # loss.extra['soft_entropy_std'] = soft_entropy.std().data[0]
        # loss.extra['soft_cross_entropy_mean'] = soft_cross_entropy.mean().data[0]
        # loss.extra['soft_cross_entropy_std'] = soft_cross_entropy.std().data[0]
        # loss.extra['soft_kl_divergence_mean'] = soft_kl_divergence.mean().data[0]
        # loss.extra['soft_kl_divergence_std'] = soft_kl_divergence.std().data[0]

        return loss
