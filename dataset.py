import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import lightning as L
import os
import cv2
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

from torchmetrics import Accuracy, F1Score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import ViTModel, ViTForImageClassification
from torchvision import models
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

class CategoryDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.img_path = df['img_path']
        self.category = df['category']
        self.brand = df["brand"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path.iloc[idx])
        img = torch.tensor(cv2.resize(img, (224, 224)), dtype=torch.float32).permute(-1, 0, 1)
        category = torch.tensor(self.category.iloc[idx], dtype=torch.long)
        brand = torch.tensor(self.brand.iloc[idx], dtype=torch.long)

        return img, category, brand

def create_loader():
    df_dir = "csv"
    df_paths = os.listdir(df_dir)

    img_dir = "image"
    df = []

    for df_path in df_paths:
        img_path = df_path.replace(".csv", "")
        df_cat = pd.read_csv(f"{df_dir}/{df_path}")
        df_cat['img_path'] = df_cat.apply(lambda x: f"{img_dir}/{img_path}/{x.name}.png" if os.path.exists(f"{img_dir}/{img_path}/{x.name}.png") else np.nan, axis=1)
        df.append(df_cat)

    df = pd.concat(df).reset_index(drop=True)
    df = df.dropna(axis=0)
    category_mapping = {"sandals": "sandals_and_flip_flops",
                    "sneakers_2": "sneakers",
                    "flip_flops": "sandals_and_flip_flops",
                    "slip_on": "slip_on_and_espadrilles",
                    "slip_on_&_espadrilles": "slip_on_and_espadrilles",
                    "sepatu_sandal_and_flipflop": "sandals_and_flip_flops",
                    "sandal_and_flip_flop_2": "sandals_and_flip_flops"}
    
    df["category"] = df["category"].apply(lambda x: category_mapping[x] if x in category_mapping.keys() else x)

    brands = [
        "Guess", "Havaianas", "Wrangler", "Dr. Martens", "ALDO", "Birkenstock",
        "CHRISTIAN LOUBOUTIN", "Clarks", "Dior", "Lacoste", "Melissa", "Nike",
        "Superdry", "VANS", "Skechers", "TIMEX", "Converse", "Wacoal",
        "Salt n Pepper", "Rockport", "Hush Puppies", "FILA", "ESPRIT", "ADIDAS",
        "Crocs", "Volcom", "Samsonite", "New Balance", "Lois Jeans", "Mango",
        "Rip Curl", "Tolliver", "Preview Itang Yunasz", "NICHOLAS EDISON",
        "Maybelline", "HUER", "Palomino", "Puma", "Reebok", "Billabong", "Levi's",
        "Yongki Komaladi", "Casio", "Alba", "3SECOND", "GREENLIGHT", "Herschel",
        "Wakai", "Tatuis", "Edifice", "Alexandre Christie", "BONIA", "Timberland",
        "Fossil", "Zalia", "ZALORA", "MANGO Man", "Aira Muslim Butik",
        "Salsabila Etnic Kebaya", "Rianty Batik", "Batik Putra Bengawan", "Rubi",
        "Cressida", "Les Catino", "Bvlgari", "Coach", "Samsung", "MICHAEL KORS",
        "Burberry", "Gucci", "Call It Spring", "Daniel Wellington",
        "Yves Saint Laurent", "Quiksilver", "Chanel", "Gobelini", "DC",
        "Love, Bonito", "bhatara batik", "Fjallraven Kanken",
        "Kamilaa by Itang Yunasz", "Keds", "Carvil", "Under Armour", "Baby-G",
        "G-Shock", "Cotton On", "Anello", "Abercrombie & Fitch", "Hollister",
        "Thule", "Lojel", "Pandora", "Avoskin", "Cool Kids", "Tory Burch",
        "New Era", "BOSS", "Polo Ralph Lauren", "MLB", "LONGCHAMP", "On",
        "BALENCIAGA", "Louis Vuitton", "Gentle Monster", "Acme De La Vie",
        "Christian Dior", "Jo Malone", "Somethinc", "2XU", "Corkcicle", "H&M",
        "MIU MIU", "Givenchy", "BOTTEGA VENETA", "Off-White", "PAYLESS",
        "LI-NING", "Smiggle", "Fred Perry", "HANASUI", "ZARA", "Roughneck 1991",
        "JACO", "MLB Korea", "Mardi Mercredi", "GENTLEWOMAN", "Loro Piana",
        "SENNHEISER", "Goyard"
    ]

    df = df[df["brand"].isin(brands)]
    df = df[df["brand"].isin((df["brand"].value_counts())[(df["brand"].value_counts() > 280)].index)]

    cat_enc = LabelEncoder()
    brand_enc = LabelEncoder()

    df["category"] = cat_enc.fit_transform(df["category"])
    df["brand"] = brand_enc.fit_transform(df["brand"])

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

    train_set = CategoryDataset(train_df)
    val_set = CategoryDataset(val_df)
    test_set = CategoryDataset(test_df)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, cat_enc, brand_enc

