import os
import pandas as pd
import random
from tqdm import tqdm

def load_csv(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist!")
        
    df = pd.read_csv(file_path)
    
    return df


def load_data(file_path):
    df = pd.read_csv(file_path)
    user_items = {row["UserId"]: [int(i) for i in row["ItemId"].split()] for _, row in df.iterrows()}
    all_items = set(item for items in user_items.values() for item in items)
    return user_items, list(all_items)


def generate_negative_samples(user_items, all_items, neg_ratio=4):
    negative_samples = {}
    all_items_set = set(all_items)

    # 總進度條
    total_samples = sum(len(items) * neg_ratio for items in user_items.values())
    progress_bar = tqdm(total=total_samples, desc="Generating negative samples", ncols=80)

    for user, pos_items in user_items.items():
        # 需要生成的負樣本數量
        num_neg = len(pos_items) * neg_ratio
        neg_items = set()

        # 以 set 差集加速查找
        available_items = list(all_items_set - set(pos_items))

        # 如果商品池很小，可能永遠找不到足夠的負樣本
        if len(available_items) < num_neg:
            print(f"⚠️ User {user} 無法找到足夠的負樣本，跳過")
            negative_samples[user] = list(available_items)
            progress_bar.update(len(available_items))
            continue

        # 直接隨機抽樣
        neg_items = random.sample(available_items, num_neg)

        # 儲存結果
        negative_samples[user] = neg_items
        
        # 更新進度
        progress_bar.update(num_neg)

    progress_bar.close()
    return negative_samples


def main():
    csv_file = "train.csv"
    
    df = load_csv(csv_file)
    # data["ItemId"] = items.split() for items in data["ItemId"]
    
    df["ItemId"] = df["ItemId"].apply(lambda x: [int(i) for i in x.split()])
    
    print(df)
        
    max_user = df["UserId"].max()
    max_item = max([max(item) for item in df["ItemId"]])
    
    user_items, all_items = load_data(csv_file)

    print(f"總共有 {len(user_items)} 位使用者")
    print(f"總共有 {len(all_items)} 個商品")

    # 生成負樣本
    negative_samples = generate_negative_samples(user_items, all_items, neg_ratio=4)

    # 確認輸出格式
    user_id = list(user_items.keys())[0]
    print(f"User {user_id} 正樣本數量：{len(user_items[user_id])}")
    print(f"User {user_id} 負樣本數量：{len(negative_samples[user_id])}")
    

if __name__ == "__main__":
    main()