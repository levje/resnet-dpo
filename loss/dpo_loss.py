import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(object):

    def __init__(self, beta=0.1):
        """
        :param beta: temperature controlling strength of KL penalty
        """
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, pi_logps, ref_logps, yw_idxs, yl_idxs):
        """
        :param pi_logprobs: current policy log probs (B,)
        :param ref_logprobs: ref policy log probs (B,)
        :param win_labels: preferred indices (T,)
        :param lose_labels: dispreferred completion indices (T,)
        :return:
        """

        pi_yw_logps, pi_yl_logps = torch.gather(pi_logps, 1, yw_idxs.unsqueeze(1)).squeeze(1), torch.gather(pi_logps, 1, yl_idxs.unsqueeze(1)).squeeze(1)
        ref_yw_logps, ref_yl_logps = torch.gather(ref_logps, 1, yw_idxs.unsqueeze(1)).squeeze(1), torch.gather(ref_logps, 1, yl_idxs.unsqueeze(1)).squeeze(1)

        pi_logratios = pi_yw_logps - pi_yl_logps
        ref_logratios = ref_yw_logps - ref_yl_logps

        losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
        rewards = self.beta * (pi_logps - ref_logps).detach()

        return losses, rewards
