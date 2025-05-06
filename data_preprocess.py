import os
import pandas as pd
import json

def load_csv(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist!")
        
    df = pd.read_csv(file_path)
    
    return df


def main():
    csv_file = "train.csv"
    
    df = load_csv(csv_file)
    # data["ItemId"] = items.split() for items in data["ItemId"]
    
    df["ItemId"] = df["ItemId"].apply(lambda x: [int(i) for i in x.split()])
        
    max_user = df["UserId"].max()
    max_item = max([max(item) for item in df["ItemId"]])
    

if __name__ == "__main__":
    main()