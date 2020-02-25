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
        self.ir_loss = torch.nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, student_net, teacher_net=None, intermediate_layers=None, cut_network=None):

        if (self.beta != 1) and not (intermediate_layers or cut_network):
            print('need intermediate layers')

        features_student, output_student = student_net(inputs, intermediate_layers, cut_network)
        if teacher_net:
            features_teacher, output_teacher = teacher_net(inputs, intermediate_layers, cut_network)

            loss_intermediate_results = 0
            for (key, value) in features_student.items():
                loss_intermediate_results += self.ir_loss(features_student[key], features_teacher[key])

            loss_knowledge_distillation = self.kd_loss(output_student, output_teacher, targets)
            total_loss = self.beta*loss_knowledge_distillation + (1-self.beta)*loss_intermediate_results
        else:
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
