import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(BPRMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        score = (user_emb * item_emb).sum(dim=1)
        return score

    def recommend(self, user_ids, top_k=50):
        # user_ids: [batch_size]
        user_emb = self.user_embedding(user_ids)  # [B, D]
        all_item_emb = self.item_embedding.weight  # [num_items, D]
        scores = torch.matmul(user_emb, all_item_emb.t())  # [B, num_items]
        _, indices = torch.topk(scores, top_k, dim=1)
        return indices  # [B, top_k]
    
   