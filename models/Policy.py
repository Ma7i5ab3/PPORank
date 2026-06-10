from __future__ import print_function
import torch
import argparse
import torch.nn as nn
import utils
from utils import AverageMeter, TqdmLoggingHandler
import Reward_utils
import numpy as np
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import logging
from torch.nn.parameter import Parameter
from models.DNN_models import *


import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
import logging


logger = logging.getLogger(__name__)


class Deep_Cross_Policy(nn.Module):

    def __init__(self, N, M, gene_dim, drug_dim, cell_dim, nlayers_cross, nlayers_deep,
                 hidden_sizes, deep_out_size, device,
                 drug_embedding=None, cell_WPmat=None,
                 train_cell=False, train_drug=False, dropout_rates=0.5,
                 cell_mean=None, drug_mean=None, overall_mean=None):

        super(Deep_Cross_Policy, self).__init__()
        self.N = N
        self.M = M
        self.drug_dim = drug_dim
        self.cell_dim = cell_dim
        self.gene_dim = gene_dim
        self.nlayers_cross = nlayers_cross
        self.nlayers_deep = nlayers_deep
        self.device = device

        self.drug_embedding = nn.Embedding(M, drug_dim, max_norm=1).float()

        # self.drug_embedding.weight.data.copy_(drug_embedding)
        if drug_embedding is not None:
            self.drug_embedding.weight.data.copy_(drug_embedding)
            self.drug_embedding.weight.requires_grad = train_drug
        else:
            nn.init.orthogonal_(self.drug_embedding.weight.data)
            self.drug_embedding.weight.requires_grad = True

        input_sizes = [self.N, self.M, self.gene_dim]

        self.cell_layer = CellNet(input_sizes, self.cell_dim, WP_pretrained=cell_WPmat).float()

        #self.cell_layer.weight.data.copy_(torch.transpose(WPmat, 0, 1))
        if cell_WPmat is not None:
            self.cell_layer.weight.data.copy_(torch.transpose(cell_WPmat, 0, 1))
            self.cell_layer.weight.requires_grad = train_cell

        self.x0_dim = drug_dim+cell_dim+1

        self.cross_classifier = CrossNet(self.x0_dim, self.nlayers_cross, drug_mean=drug_mean).float()
        #self.deep_classifier = build_mlp(self.total_dim, deep_out_size, deep_layers, hidden_sizes).double()
        self.deep_classifier = DeepNet(self.x0_dim, deep_out_size, nlayers_deep,
                                       hidden_sizes, dropout_rates=dropout_rates).float()

        #self.cross_classifier = build_cross(self.total_dim,cross_layers)
        # self.classifierC0 = CrossLayer(self.total_dim).double()  # if we only consider the level 1 cross, to save
        # parameters, only use one single cross layer, input dim=output dim =
        #self.classifierC1 = CrossLayer(self.total_dim).double()

        in_final_dim = deep_out_size + self.x0_dim + 1  # 1 is for cosine similarity
        self.activation = nn.ReLU().float()
        self.BN = nn.BatchNorm1d(M).float()
        #self.drop_out = nn.Dropout(p=0.2)
        # input 3-D, so output_size=1 in the 3rd dim
        self.classifierF = nn.Linear(in_final_dim, 1, bias=True).float()
        nn.init.constant_(self.classifierF.bias, 10.0)
        if overall_mean:
            self.classifierF.bias.data.copy_(overall_mean)

    def forward(self, input, filter_masks=None):
        # input:[B,M,cell_dim+1],filter_masks [B,M,1]
        B, M, _ = input.size()
        device = input.device
        filter_masks = torch.ones(B, M, 1, device=device) if filter_masks is None \
                       else filter_masks.view(B, M, 1).to(device)

        # cell features are identical for all M drugs per cell — compute once and expand
        cell_emb = self.cell_layer(input[:, 0, :-1]).unsqueeze(1).expand(-1, M, -1)  # (B,M,f)
        drug_emb = self.drug_embedding(input[:, :, -1].long()) * filter_masks  # (B,M,f)

        cos1 = (cell_emb * drug_emb).sum(2).view(B, self.M, 1)
        x0 = torch.cat((cell_emb, drug_emb, cos1), 2)

        D_out = self.deep_classifier(x0)
        C_out = self.cross_classifier(x0.view(-1, x0.size()[-1])).view(B, self.M, -1)

        in_final = self.activation(torch.cat((D_out, C_out, cos1), 2))
        in_final = self.BN(in_final)
        out = self.classifierF(in_final)
        return out, in_final

    def forward_from_obs(self, obs_actor, drug_inds):
        """Efficient path for evaluate_actions: skips building the (B,M,P+1) input tensor.
        filter_masks is not applied to drug_emb here — consistent with the normal forward()
        path which is always called without filter_masks (defaults to ones, no effect)."""
        B, M = drug_inds.shape
        cell_emb = self.cell_layer(obs_actor).unsqueeze(1).expand(-1, M, -1)  # (B,M,f)
        drug_emb = self.drug_embedding(drug_inds)  # (B,M,f)

        cos1 = (cell_emb * drug_emb).sum(2).view(B, self.M, 1)
        x0 = torch.cat((cell_emb, drug_emb, cos1), 2)

        D_out = self.deep_classifier(x0)
        C_out = self.cross_classifier(x0.view(-1, x0.size()[-1])).view(B, self.M, -1)

        in_final = self.activation(torch.cat((D_out, C_out, cos1), 2))
        in_final = self.BN(in_final)
        return self.classifierF(in_final), in_final

    def pred_value(self, input, true_scores, filter_masks, cur_ranks):
        # input [B,M,cell_dim+1], filter_masks: [B,M],cur_ranks[B,1]
        scores = self.forward(input).squeeze()  # [N,M]
        B = scores.size()[0]
        ndcg_vals = Reward_utils.ndcg_from_cur_pos(true_scores.numpy(),
                                                   scores.cpu().detach().numpy(),
                                                   filter_masks.cpu().detach().numpy(),
                                                   cur_ranks)
        return ndcg_vals


class ValueNet(nn.Module):
    def __init__(self, input_size, output_size, n_layers, hidden_sizes):
        super(ValueNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        self.mlp = build_mlp(input_size, output_size, n_layers, hidden_sizes).float()

    def forward(self, input):
        """
        the input should be the current state:
        the remaining all documents:N*1*P
        """

        values = self.mlp(input.float())
        # return self.softmax(values)  # [0,1]
        return torch.sigmoid(values)


class ConvValueNet(nn.Module):
    def __init__(self, input_c, output_c, kernel_size):
        super(ConvValueNet, self).__init__()
        self.conv1d_layer = nn.Conv1d(input_c, output_c, kernel_size).float()
        self.pooling = nn.AvgPool1d(kernel_size)

    def forward(self, input):
        out = self.conv1d_layer(input)
        out = self.pooling(out)
        return torch.sigmoid(out)


class PolicyGradientRank(nn.Module):
    def __init__(self, P):  # m is the feature dimension except x0 term
        super(PolicyGradientRank, self).__init__()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, P),
                               stride=(1, P), bias=True)
        self.conv2.weight.data.zero_()
        self.softmax = nn.Softmax()

    def forward(self, input):  # online ranking
        return self.conv2(input)


class PPO_Policy(nn.Module):
    def __init__(self, N, M, gene_dim, drug_dim, drug_pretrained_weight, cell_dim, WPmat,
                 nlayers_cross, nlayers_deep, hidden_sizes, deep_out_size,
                 value_n_layers, value_hidden_sizes, device,
                 train_cell=False, train_drug=False, drug_mean=None, overall_mean=None):
        super(PPO_Policy, self).__init__()
        self.M = M
        self.N = N

        self.actor = Deep_Cross_Policy(
            N, M, gene_dim, drug_dim, cell_dim, nlayers_cross, nlayers_deep,
            hidden_sizes, deep_out_size, device, drug_embedding=drug_pretrained_weight,
            cell_WPmat=WPmat, train_cell=train_cell, train_drug=train_drug,
            drug_mean=torch.tensor(drug_mean).float(), overall_mean=torch.tensor(overall_mean).float())

        self.critic_size = self.actor.cell_dim + +self.actor.drug_dim + 1 + deep_out_size + 1
        #self.critic_size = self.actor.cell_dim + M*(self.actor.drug_dim+1)
        # input consits of cell features and drug features and cos similarity
        # can be done through actor and then using masks then concatenate
        # the output size should be 1,which is the state
        # self.critic_old = ValueNet(self.critic_size, 1, value_n_layers, value_hidden_sizes)
        self.critic = ConvValueNet(M, 1, cell_dim)

        self.parameters = list(self.actor.parameters())+list(self.critic.parameters())

    def forward(self, input, filter_masks=None):
        # the forward part will output the scores from actor policy,which is M dimension
        actor_output = self.actor(input, filter_masks=filter_masks)
        return actor_output

    def get_fts_vecs(self, input, masks):
        """
        #input:[B,M,P1+1], masks: (B,M)
        mainly used to build a single state vector by concating cell-line, candidate drugs, cos-sim
        into a single vector, output size as (B,cell_dim+M*(drug_dim+1))
        """

        masks = masks + 1.0
        masks[masks == float('-inf')] = 0.0
        if len(input.size()) < 3:
            input = input.unsqueeze(0)
        B = input.size()[0]
        cell_fts = input.clone()[:, :, :-1]
        drug_index = input.clone()[:, :, -1].long()  # all drugs corresponding to the cell-line

        cell_emb = self.actor.cell_layer(cell_fts)  # (B,M,f)
        cell_fts = cell_emb[:, 0, :].view(B, -1)  # (B,P1)

        drug_emb = self.actor.drug_embedding(drug_index)  # (B,M,f)
        masks = masks.unsqueeze(-1).expand_as(drug_emb)  # masks have change from [B,M], then expand to [B,M,P2]
        drug_emb = torch.mul(drug_emb, masks)
        drug_fts = drug_emb.view(B, -1)  # (B,M*P2)

        cos1 = (cell_emb * drug_emb).sum(2).view(B, self.M, 1).squeeze(-1)  # (B,M)
        all_fts = torch.cat((cell_fts, drug_fts, cos1), 1)  # (B,cell_dim+M*(drug_dim+1))
        return all_fts

    def get_value(self, input, filter_masks):
        # this ouput the value from critic the features include cell-line,drug and cosimine
        # input [B,M,cell_dim+1(drug_index)], filter_masks:[B,M], for each cellline in B batches,
        # drug fts include all candidate drugs
        if input.size()[-1] != self.critic_size:
            value_net_input = self.get_fts_vecs(input, filter_masks)  # (B,cell_dim+M*(drug_dim+1))
            critic_value = self.critic_old(value_net_input)
        else:
            critic_value = self.critic_old(input)
        return critic_value

    def get_value_from_actor(self, input):
        critic_value = self.critic(input)
        return critic_value

    def sample_action(self, scores, filter_masks):
        dimension = len(scores.shape)-1

        probs = nn.functional.log_softmax(scores+filter_masks, dim=dimension).exp()  # (B,M) or (M)
        #torch.nan_to_num(probs, nan=0.0)
        end_masks = torch.any(probs.isnan(), dim=dimension)  # (B,1)
        probs.nan_to_num_(nan=1e-8)

        selected_drug_id = torch.multinomial(probs, 1).T[0]  # (B)
        selected_drug_id[end_masks] = 1
        return selected_drug_id, end_masks

    def get_log_prob(self, scores, filter_masks, selected_drug_ids):
        self.dist = torch.distributions.Categorical(logits=scores+filter_masks)
        log_prob = self.dist.log_prob(selected_drug_ids)  # .view(1, -1))
        probs = self.dist.probs
        dist_entropy = -(scores*probs).sum(-1)
        return log_prob.view(-1, 1), dist_entropy.mean().view(-1, 1)

    def act(self, input, filter_masks):
        # input:[B,M,cell_dim+1]
        # masks:[B,M]
        scores = self.actor(input)
        action = self.sample_action(scores, filter_masks)
        action_log_prob, dist_entropy = self.get_log_prob(scores, filter_masks, action)
        filter_masks[:, action] = float('-inf')

        return scores, action, action_log_prob, filter_masks

    def evaluate_actions(self, obs_actor, filter_masks, actions):
        device = actions.device
        B = obs_actor.size(0)
        M = filter_masks.size(1)

        drug_inds = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)  # (B,M), no numpy
        scores, critic_input = self.actor.forward_from_obs(obs_actor, drug_inds)
        scores = scores.squeeze()
        values = self.critic(critic_input)
        action_log_prob, dist_entropy = self.get_log_prob(scores, filter_masks, actions.squeeze(-1))
        return values, action_log_prob, dist_entropy


class PPO_Shared_Policy(nn.Module):
    def __init__(self, N, M, gene_dim, drug_dim, drug_pretrained_weight, cell_dim, WPmat,
                 cross_layers, deep_layers, hidden_sizes, deep_out_size,
                 value_n_layers, value_hidden_sizes,
                 train_cell=False, train_drug=False):

        super(PPO_Policy, self).__init__()
        self.M = M
        self.N = N

        self.actor = Deep_Cross_Policy(
            N, M, gene_dim, drug_dim, drug_pretrained_weight, cell_dim, WPmat,
            cross_layers, deep_layers, hidden_sizes, deep_out_size,
            train_cell=train_cell, train_drug=train_drug)
        self.params = list(self.actor.parameters())
        #self.critic_size = self.actor.cell_dim + M*(self.actor.drug_dim+1)
        # input consits of cell features and drug features and cos similarity
        # can be done through actor and then using masks then concatenate
        # the output size should be 1
        self.critic = self.actor

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        # the forward part will output the scores from actor policy,which is M dimension
        actor_output = self.actor(input)
        return actor_output

    def get_fts_vecs(self, input, masks):
        # input:[B,M,P1+1], masks: (B,M)
        masks = masks + 1.0
        masks[masks == float('-inf')] = 0.0
        if len(input.size()) < 3:
            input = input.unsqueeze(0)
        B = input.size()[0]
        cell_fts = input[:, :, :-1]
        drug_index = input[:, :, -1].type(torch.LongTensor)  # all drugs corresponding to the cell-line

        cell_emb = self.actor.cell_layer(cell_fts)  # (B,M,f)
        cell_fts = cell_emb[:, 0, :].view(B, -1)  # (B,P1)

        drug_emb = self.actor.drug_embedding(drug_index)  # (B,M,f)
        masks = masks.unsqueeze(-1).expand_as(drug_emb)
        drug_emb = torch.mul(drug_emb, masks)
        drug_fts = drug_emb.view(B, -1)  # (B,M*P2)

        cos1 = (cell_emb * drug_emb).sum(2).view(B, self.M, 1).squeeze(-1)  # (B,M)
        all_fts = torch.cat((cell_fts, drug_fts, cos1), 1)  # (B,cell_dim+M*(drug_dim+1))
        return all_fts

    def get_value(self, input, true_scores, filter_masks, cur_ranks):
        # this ouput the value from critic the features include cell-line,drug and cosimine
        # input [B,M,cell_dim+1(drug_index)], filter_masks:[B,M], for each cellline in B batches,
        # drug fts include all candidate drugs

        critic_value = self.critic.pred_value(input, true_scores, filter_masks, cur_ranks)
        return critic_value

    def sample_action(self, scores, filter_masks):

        probs = nn.functional.log_softmax(scores+filter_masks, dim=0).exp()
        selected_drug_id = torch.multinomial(probs, 1)[0]
        return selected_drug_id

    def get_log_prob(self, scores, filter_masks, selected_drug_ids):
        self.dist = torch.distributions.Categorical(logits=scores+filter_masks)
        log_prob = self.dist.log_prob(selected_drug_ids.view(1, -1))
        probs = self.dist.probs
        dist_entropy = -(scores*probs).sum(-1)
        return log_prob.view(-1, 1), dist_entropy.mean()

    def act(self, input, filter_masks):
        # input:[B,M,cell_dim+1]
        # masks:[B,M]
        scores = self.actor(input)
        action = self.sample_action(scores, filter_masks)
        action_log_prob, dist_entropy = self.get_log_prob(scores, filter_masks, action)
        filter_masks[:, action] = float('-inf')

        return scores, action, action_log_prob, filter_masks

    def evaluate_actions(self, obs_critic, obs_actor, filter_masks, actions, true_scores, cur_ranks):
        # here the values should requires grad true
        B = obs_actor.size()[0]
        M = filter_masks.size()[1]
        #obs_actor_np = obs_actor.cpu().data.numpy()
        drug_inds = torch.from_numpy(np.arange(M).reshape(1, M).repeat(B, axis=0).reshape(B, M, 1))
        cell_fts = torch.repeat_interleave(obs_actor.view(B, 1, -1), repeats=M, dim=1).float()
        # np.repeat(obs_actor_np[:,np.newaxis,:],M,axis=1)
        input = torch.cat((cell_fts, drug_inds.float()), axis=2)

        scores = self.actor(input).squeeze()
        values = self.get_value(input, true_scores, filter_masks, cur_ranks)

        action_log_prob, dist_entropy = self.get_log_prob(scores, filter_masks, actions)

        return values, action_log_prob, dist_entropy
