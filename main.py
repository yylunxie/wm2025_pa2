import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 設定使用的 GPU 編號
import torch
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
from utils import mapk
import torch.utils.data as data
from model_bpr import BPRMF
from dataset import PositivePairDataset
from collections import defaultdict


def generate_submission(model, dataset, train_data, output_path, device, top_k=50, rec_pool=1000):
    model.eval()
    user_ids = sorted(dataset.keys())
    all_predictions = []

    with torch.no_grad():
        for user in tqdm(user_ids):
            user_tensor = torch.LongTensor([user]).to(device)
            rec_items = model.recommend(user_tensor, top_k=rec_pool)[0].tolist()

            seen_items = set(train_data[user])
            final_rec = [i for i in rec_items if i not in seen_items][:top_k]

            # 避免不足 50 個
            if len(final_rec) < top_k:
                print(f"⚠️ User {user} only has {len(final_rec)} unseen predictions.")
                padding_item = max([i for items in train_data.values() for i in items], default=0) + 1
                while len(final_rec) < top_k:
                    final_rec.append(padding_item)

            all_predictions.append({
                "UserId": user,
                "ItemId": " ".join(map(str, final_rec))
            })

    pd.DataFrame(all_predictions).to_csv(output_path, index=False)
    print(f"✅ Submission saved to {output_path}")


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def split_data(user_items, ratio):
    train_data = {}
    valid_data = {}

    for user, items in user_items.items():
        if len(items) < 2:
            train_data[user] = items
            valid_data[user] = []
            continue
        split_point = int(len(items) * ratio)
        # split_point = -10
        train_data[user] = items[:split_point]
        valid_data[user] = items[split_point:]
    return train_data, valid_data
            

def evaluate_model(model, train_data, val_data, device, top_k=50):
    model.eval()
    all_preds = []
    all_truth = []

    with torch.no_grad():
        for user in sorted(val_data.keys()):
            true_items = val_data[user]
            if not true_items:
                continue  # 無驗證資料的 user 跳過

            user_tensor = torch.LongTensor([user]).to(device)
            rec_items = model.recommend(user_tensor, top_k=top_k)[0].tolist()

            seen_items = set(train_data[user])
            final_rec = [i for i in rec_items if i not in seen_items][:top_k]

            # 補滿推薦長度避免報錯
            while len(final_rec) < top_k:
                final_rec.append(0)

            all_preds.append(final_rec)
            all_truth.append(true_items)

    return mapk(all_truth, all_preds, k=top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train.csv")
    parser.add_argument("--output_path", type=str, default="output/bpr_submission.csv")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--neg", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model", action="store_true", help="Save model after training")
    args = parser.parse_args()
    best_score = 0
    best_epoch = -1
    
    set_seed(42)
    save_dir = f"weights/d{args.embedding_dim}"
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the dataset
    user_items = defaultdict(list)
    user_set = set()
    item_set = set()
    # Read the train.csv file
    # train.csv 的格式是 user_id, item_id1 item_id2 item_id3 ...
    
    with open(args.train_path, 'r') as f:
        next(f)  # 跳過第一行的表頭
        for line in f:
            parts = line.strip().split(',')
            user = int(parts[0])
            items = list(map(int, parts[1].split()))
            user_items[user] = items
            user_set.add(user)
            item_set.update(items)
    all_items = list(item_set)
    train_data, valid_data = split_data(user_items, 0.8)
    train_complement = {user: [item for item in all_items if item not in items] for user, items in train_data.items()}
    # train_complement 是一個字典，key 是 user id，value 是該 user 沒有看過的 item id list
    valid_complement = {user: [item for item in all_items if item not in items] for user, items in valid_data.items()}
    dataset = PositivePairDataset(train_data)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size)

    # dataset 會將 train.csv 讀進來並做資料前處理，
    # 包含 train_data, valid_data, user_items, user_set, item_set
    # train_data 是一個字典，key 是 user id，value 是該 user 看過的 item id list
    # valid_data 是一個字典，key 是 user id，value 是該 user 在驗證集上看過的 item id list
    # user_items 是一個字典，key 是 user id，value 是該 user 看過的 item id list
    # user_set 和 item_set 是所有 user 和 item 的集合
    num_users = max(user_set) + 1
    num_items = max(item_set) + 1
    
    if args.mode == "train":
        model = BPRMF(num_users, num_items, embedding_dim=args.embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        for epoch in range(args.epochs):

            total_loss = 0
            for batch_users, batch_pos_items in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # for batch in dataloader:
                batch_users = batch_users.to(device)
                batch_pos_items = batch_pos_items.to(device)
                
                triplet_users = []
                triplet_pos = []
                triplet_neg = []
                
                for i in range(len(batch_users)):
                    u = batch_users[i].item()
                    pos = batch_pos_items[i].item()
                    
                    seen_items = set(user_items[u])
                    candidates = [item for item in all_items if item not in seen_items]
                    if len(candidates) == 0:
                        continue
                    candidates = random.sample(candidates, min(100, len(candidates)))
                    
                    u_tensor = torch.LongTensor([u] * len(candidates)).to(device)  # [C]
                    item_tensor = torch.LongTensor(candidates).to(device)          # [C]
                    
                    # candidate = random.sample(train_complement[int(user[i])], 200)
                    scores = model(u_tensor, item_tensor).detach()  # [C]
                    top_indices = torch.topk(scores, k=min(10, len(candidates))).indices
                    hard_negatives = [candidates[idx] for idx in top_indices.tolist()]
                    final_negs = random.sample(hard_negatives, min(args.neg, len(hard_negatives)))
                    for neg in final_negs:
                        triplet_users.append(u)
                        triplet_pos.append(pos)
                        triplet_neg.append(neg)
                        
                if not triplet_users:
                    continue
                
                u_tensor = torch.LongTensor(triplet_users).to(device)
                i_tensor = torch.LongTensor(triplet_pos).to(device)
                j_tensor = torch.LongTensor(triplet_neg).to(device)
                
                u_emb = model.user_embedding(u_tensor)
                i_emb = model.item_embedding(i_tensor)
                j_emb = model.item_embedding(j_tensor)
                
                pos_scores = (u_emb * i_emb).sum(dim=1)
                neg_scores = (u_emb * j_emb).sum(dim=1)
                
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
            print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

            val_map = evaluate_model(model, val_data=valid_data, train_data=train_data, device=device)
            print(f"[Validation MAP@50]: {val_map:.4f}")

            if val_map > best_score:
                best_score = val_map
                best_epoch = epoch
                model_path = f"weights/d{args.embedding_dim}/best_model_bpr.pt"
                torch.save(model.state_dict(), model_path)
                print(f"✅ New best model saved at epoch {epoch+1} with MAP@50 = {val_map:.4f}")
    # elif args.mode == "val":
    model = BPRMF(num_users, num_items, embedding_dim=args.embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    generate_submission(model, user_items, train_data=train_data, output_path=args.output_path, device=device)
    print(f"🏁 Best MAP@50 = {best_score:.4f} at epoch {best_epoch+1}")
    # Log results
    with open("dimension_results.csv", "a") as f:
        if not os.path.exists("dimension_results.csv") or os.path.getsize("dimension_results.csv") == 0:
            f.write("embedding_dim,best_map@50,best_epoch\n")
        f.write(f"{args.embedding_dim},{best_score:.4f},{best_epoch+1}\n")
    
    print(f"Results for dimension {args.embedding_dim} added to dimension_results.csv")
    

if __name__ == "__main__":
    main()