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

class MultiLabelHead(nn.Module):
    def __init__(self, in_features, n_category, n_brand):
        super(MultiLabelHead, self).__init__()
        self.n_category = n_category
        self.n_brand = n_brand

        self.fc_cat = nn.Linear(in_features, n_category)
        self.fc_brand = nn.Linear(in_features, n_category)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        input = self.dropout(input)
        cat_logits = self.fc_cat(input)
        brand_logits = self.fc_brand(input)

        return cat_logits, brand_logits

class ResNet101(L.LightningModule):
    def __init__(self, n_category, n_brand):
        super(ResNet101, self).__init__()
        self.save_hyperparameters()
        self.n_category = n_category
        self.n_brand = n_brand

        self.resnet = models.resnet101(weights=models.ResNet101_Weights)
        self.head = MultiLabelHead(self.resnet.fc.in_features, self.n_category, self.n_brand)
        self.resnet.fc = self.head
        self.accuracy_cat = Accuracy(task="multiclass", num_classes=self.n_category)
        self.accuracy_brand = Accuracy(task="multiclass", num_classes=self.n_brand)
        self.f1_cat = F1Score(task="multiclass", num_classes=self.n_category, average="macro")
        self.f1_brand = F1Score(task="multiclass", num_classes=self.n_brand, average="macro")
        self.criterion_cat = nn.CrossEntropyLoss()
        self.criterion_brand = nn.CrossEntropyLoss()

        self.train_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.val_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.test_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}

    def forward(self, pixel_values):
        cat_logits, brand_logits = self.resnet(pixel_values)

        return cat_logits, brand_logits
    
    def _common_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)
        loss_cat = self.criterion_cat(cat_pred, category)
        loss_brand = self.criterion_cat(brand_pred, brand)
        loss = loss_cat + loss_brand

        return loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand
    
    def training_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cat_loss', loss_cat)
        self.log('train_brand_loss', loss_brand)

        self.train_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.train_output["cat_actual"].append(category)
        self.train_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.train_output["brand_actual"].append(brand)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.val_output["cat_actual"].append(category)
        self.val_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.val_output["brand_actual"].append(brand)

        return self.val_output
    
    def test_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        self.test_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.test_output["cat_actual"].append(category)
        self.test_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.test_output["brand_actual"].append(brand)
        return self.test_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer
                ),
                "monitor": "val_loss",
            }
        }
    
    def predict_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)

        return cat_pred, brand_pred
    
    def on_train_epoch_end(self):
        cat_pred = torch.cat(self.train_output["cat_pred"])
        cat_actual = torch.cat(self.train_output["cat_actual"])
        brand_pred = torch.cat(self.train_output["brand_pred"])
        brand_actual = torch.cat(self.train_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        self.log("train_cat_f1", cat_f1_score)
        self.log("train_cat_accuracy", cat_accuracy_score)
        self.log("train_brand_f1", brand_f1_score)
        self.log("train_brand_accuracy", brand_accuracy_score)
        self.log("train_exact_match_ratio", exact_match_ratio)

        self.train_output["cat_pred"].clear()
        self.train_output["cat_actual"].clear()
        self.train_output["brand_pred"].clear()
        self.train_output["brand_actual"].clear()
    
    def on_validation_epoch_end(self):
        cat_pred = torch.cat(self.val_output["cat_pred"])
        cat_actual = torch.cat(self.val_output["cat_actual"])
        brand_pred = torch.cat(self.val_output["brand_pred"])
        brand_actual = torch.cat(self.val_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        print(f'\nCategory f1_score: {cat_f1_score} | Category accuracy: {cat_accuracy_score}')
        print(f'Brand f1_score: {brand_f1_score} | Brand accuracy: {brand_accuracy_score}')
        print(f'Exact match ratio: {exact_match_ratio}')

        self.log("val_cat_f1", cat_f1_score)
        self.log("val_cat_accuracy", cat_accuracy_score)
        self.log("val_brand_f1", brand_f1_score)
        self.log("val_brand_accuracy", brand_accuracy_score)
        self.log("val_exact_match_ratio", exact_match_ratio)

        self.val_output["cat_pred"].clear()
        self.val_output["cat_actual"].clear()
        self.val_output["brand_pred"].clear()
        self.val_output["brand_actual"].clear()

    def on_test_epoch_end(self):
        cat_pred = torch.cat(self.test_output["cat_pred"])
        cat_actual = torch.cat(self.test_output["cat_actual"])
        brand_pred = torch.cat(self.test_output["brand_pred"])
        brand_actual = torch.cat(self.test_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)

        cat_pred = cat_pred.tolist()
        cat_actual = cat_actual.tolist()
        brand_pred = brand_pred.tolist()
        brand_actual = brand_actual.tolist()

        exact_match_ratio = np.mean((np.array(cat_pred) == np.array(cat_actual)) & (np.array(brand_pred) == np.array(brand_actual)))
        self.log("test_cat_f1", cat_f1_score)
        self.log("test_cat_accuracy", cat_accuracy_score)
        self.log("test_brand_f1", brand_f1_score)
        self.log("test_brand_accuracy", brand_accuracy_score)
        self.log("test_exact_match_ratio", exact_match_ratio)
        print("Exact match ratio :", exact_match_ratio)
        
        print('\nCategory Metrics')
        print(classification_report(cat_actual, cat_pred))
        cm = confusion_matrix(cat_actual, cat_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()
        print('\nBrand Metrics')
        print(classification_report(brand_actual, brand_pred))
        cm = confusion_matrix(brand_actual, brand_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()

        self.test_output["cat_pred"].clear()
        self.test_output["cat_actual"].clear()
        self.test_output["brand_pred"].clear()
        self.test_output["brand_actual"].clear()

class ResNet50(L.LightningModule):
    def __init__(self, n_category, n_brand):
        super(ResNet50, self).__init__()
        self.save_hyperparameters()
        self.n_category = n_category
        self.n_brand = n_brand

        self.resnet = models.resnet50(weights=models.ResNet50_Weights)
        self.head = MultiLabelHead(self.resnet.fc.in_features, self.n_category, self.n_brand)
        self.resnet.fc = self.head
        self.accuracy_cat = Accuracy(task="multiclass", num_classes=self.n_category)
        self.accuracy_brand = Accuracy(task="multiclass", num_classes=self.n_brand)
        self.f1_cat = F1Score(task="multiclass", num_classes=self.n_category, average="macro")
        self.f1_brand = F1Score(task="multiclass", num_classes=self.n_brand, average="macro")
        self.criterion_cat = nn.CrossEntropyLoss()
        self.criterion_brand = nn.CrossEntropyLoss()

        self.train_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.val_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.test_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}

    def forward(self, pixel_values):
        cat_logits, brand_logits = self.resnet(pixel_values)

        return cat_logits, brand_logits
    
    def _common_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)
        loss_cat = self.criterion_cat(cat_pred, category)
        loss_brand = self.criterion_cat(brand_pred, brand)
        loss = loss_cat + loss_brand

        return loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand
    
    def training_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cat_loss', loss_cat)
        self.log('train_brand_loss', loss_brand)

        self.train_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.train_output["cat_actual"].append(category)
        self.train_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.train_output["brand_actual"].append(brand)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.val_output["cat_actual"].append(category)
        self.val_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.val_output["brand_actual"].append(brand)

        return self.val_output
    
    def test_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        self.test_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.test_output["cat_actual"].append(category)
        self.test_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.test_output["brand_actual"].append(brand)
        return self.test_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer
                ),
                "monitor": "val_loss",
            }
        }
    
    def predict_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)

        return cat_pred, brand_pred
    
    def on_train_epoch_end(self):
        cat_pred = torch.cat(self.train_output["cat_pred"])
        cat_actual = torch.cat(self.train_output["cat_actual"])
        brand_pred = torch.cat(self.train_output["brand_pred"])
        brand_actual = torch.cat(self.train_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        self.log("train_cat_f1", cat_f1_score)
        self.log("train_cat_accuracy", cat_accuracy_score)
        self.log("train_brand_f1", brand_f1_score)
        self.log("train_brand_accuracy", brand_accuracy_score)
        self.log("train_exact_match_ratio", exact_match_ratio)

        self.train_output["cat_pred"].clear()
        self.train_output["cat_actual"].clear()
        self.train_output["brand_pred"].clear()
        self.train_output["brand_actual"].clear()
    
    def on_validation_epoch_end(self):
        cat_pred = torch.cat(self.val_output["cat_pred"])
        cat_actual = torch.cat(self.val_output["cat_actual"])
        brand_pred = torch.cat(self.val_output["brand_pred"])
        brand_actual = torch.cat(self.val_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        print(f'\nCategory f1_score: {cat_f1_score} | Category accuracy: {cat_accuracy_score}')
        print(f'Brand f1_score: {brand_f1_score} | Brand accuracy: {brand_accuracy_score}')
        print(f'Exact match ratio: {exact_match_ratio}')

        self.log("val_cat_f1", cat_f1_score)
        self.log("val_cat_accuracy", cat_accuracy_score)
        self.log("val_brand_f1", brand_f1_score)
        self.log("val_brand_accuracy", brand_accuracy_score)
        self.log("val_exact_match_ratio", exact_match_ratio)

        self.val_output["cat_pred"].clear()
        self.val_output["cat_actual"].clear()
        self.val_output["brand_pred"].clear()
        self.val_output["brand_actual"].clear()

    def on_test_epoch_end(self):
        cat_pred = torch.cat(self.test_output["cat_pred"])
        cat_actual = torch.cat(self.test_output["cat_actual"])
        brand_pred = torch.cat(self.test_output["brand_pred"])
        brand_actual = torch.cat(self.test_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)

        cat_pred = cat_pred.tolist()
        cat_actual = cat_actual.tolist()
        brand_pred = brand_pred.tolist()
        brand_actual = brand_actual.tolist()

        exact_match_ratio = np.mean((np.array(cat_pred) == np.array(cat_actual)) & (np.array(brand_pred) == np.array(brand_actual)))
        self.log("test_cat_f1", cat_f1_score)
        self.log("test_cat_accuracy", cat_accuracy_score)
        self.log("test_brand_f1", brand_f1_score)
        self.log("test_brand_accuracy", brand_accuracy_score)
        self.log("test_exact_match_ratio", exact_match_ratio)

        print("Exact match ratio :", exact_match_ratio)
        
        print('\nCategory Metrics')
        print(classification_report(cat_actual, cat_pred))
        cm = confusion_matrix(cat_actual, cat_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()
        print('\nBrand Metrics')
        print(classification_report(brand_actual, brand_pred))
        cm = confusion_matrix(brand_actual, brand_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()

        self.test_output["cat_pred"].clear()
        self.test_output["cat_actual"].clear()
        self.test_output["brand_pred"].clear()
        self.test_output["brand_actual"].clear()

class VIT(L.LightningModule):
    def __init__(self, n_category, n_brand):
        super(VIT, self).__init__()
        self.save_hyperparameters()
        self.n_category = n_category
        self.n_brand = n_brand

        self.vit = models.vit_b_32(weights=models.ViT_B_32_Weights)
        self.head = MultiLabelHead(self.vit.heads.head.in_features, self.n_category, self.n_brand)
        self.vit.heads.head = self.head
        self.accuracy_cat = Accuracy(task="multiclass", num_classes=self.n_category)
        self.accuracy_brand = Accuracy(task="multiclass", num_classes=self.n_brand)
        self.f1_cat = F1Score(task="multiclass", num_classes=self.n_category, average="macro")
        self.f1_brand = F1Score(task="multiclass", num_classes=self.n_brand, average="macro")
        self.criterion_cat = nn.CrossEntropyLoss()
        self.criterion_brand = nn.CrossEntropyLoss()

        self.train_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.val_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.test_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}

    def forward(self, pixel_values):
        cat_logits, brand_logits = self.vit(pixel_values)

        return cat_logits, brand_logits
    
    def _common_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)
        loss_cat = self.criterion_cat(cat_pred, category)
        loss_brand = self.criterion_cat(brand_pred, brand)
        loss = loss_cat + loss_brand

        return loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand
    
    def training_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cat_loss', loss_cat)
        self.log('train_brand_loss', loss_brand)

        self.train_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.train_output["cat_actual"].append(category)
        self.train_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.train_output["brand_actual"].append(brand)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.val_output["cat_actual"].append(category)
        self.val_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.val_output["brand_actual"].append(brand)

        return self.val_output
    
    def test_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        self.test_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.test_output["cat_actual"].append(category)
        self.test_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.test_output["brand_actual"].append(brand)
        return self.test_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer
                ),
                "monitor": "val_loss",
            }
        }
    
    def predict_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)

        return cat_pred, brand_pred
    
    def on_train_epoch_end(self):
        cat_pred = torch.cat(self.train_output["cat_pred"])
        cat_actual = torch.cat(self.train_output["cat_actual"])
        brand_pred = torch.cat(self.train_output["brand_pred"])
        brand_actual = torch.cat(self.train_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        self.log("train_cat_f1", cat_f1_score)
        self.log("train_cat_accuracy", cat_accuracy_score)
        self.log("train_brand_f1", brand_f1_score)
        self.log("train_brand_accuracy", brand_accuracy_score)
        self.log("train_exact_match_ratio", exact_match_ratio)

        self.train_output["cat_pred"].clear()
        self.train_output["cat_actual"].clear()
        self.train_output["brand_pred"].clear()
        self.train_output["brand_actual"].clear()
    
    def on_validation_epoch_end(self):
        cat_pred = torch.cat(self.val_output["cat_pred"])
        cat_actual = torch.cat(self.val_output["cat_actual"])
        brand_pred = torch.cat(self.val_output["brand_pred"])
        brand_actual = torch.cat(self.val_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        print(f'\nCategory f1_score: {cat_f1_score} | Category accuracy: {cat_accuracy_score}')
        print(f'Brand f1_score: {brand_f1_score} | Brand accuracy: {brand_accuracy_score}')
        print(f'Exact match ratio: {exact_match_ratio}')

        self.log("val_cat_f1", cat_f1_score)
        self.log("val_cat_accuracy", cat_accuracy_score)
        self.log("val_brand_f1", brand_f1_score)
        self.log("val_brand_accuracy", brand_accuracy_score)
        self.log("val_exact_match_ratio", exact_match_ratio)

        self.val_output["cat_pred"].clear()
        self.val_output["cat_actual"].clear()
        self.val_output["brand_pred"].clear()
        self.val_output["brand_actual"].clear()

    def on_test_epoch_end(self):
        cat_pred = torch.cat(self.test_output["cat_pred"])
        cat_actual = torch.cat(self.test_output["cat_actual"])
        brand_pred = torch.cat(self.test_output["brand_pred"])
        brand_actual = torch.cat(self.test_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)

        cat_pred = cat_pred.tolist()
        cat_actual = cat_actual.tolist()
        brand_pred = brand_pred.tolist()
        brand_actual = brand_actual.tolist()

        exact_match_ratio = np.mean((np.array(cat_pred) == np.array(cat_actual)) & (np.array(brand_pred) == np.array(brand_actual)))
        self.log("test_cat_f1", cat_f1_score)
        self.log("test_cat_accuracy", cat_accuracy_score)
        self.log("test_brand_f1", brand_f1_score)
        self.log("test_brand_accuracy", brand_accuracy_score)
        self.log("test_exact_match_ratio", exact_match_ratio)

        print("Exact match ratio :", exact_match_ratio)
        
        print('\nCategory Metrics')
        print(classification_report(cat_actual, cat_pred))
        cm = confusion_matrix(cat_actual, cat_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()
        print('\nBrand Metrics')
        print(classification_report(brand_actual, brand_pred))
        cm = confusion_matrix(brand_actual, brand_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()

        self.test_output["cat_pred"].clear()
        self.test_output["cat_actual"].clear()
        self.test_output["brand_pred"].clear()
        self.test_output["brand_actual"].clear()
  
class CONVNEXT(L.LightningModule):
    def __init__(self, n_category, n_brand):
        super(CONVNEXT, self).__init__()
        self.save_hyperparameters()
        self.n_category = n_category
        self.n_brand = n_brand

        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.convnext = models.convnext_tiny(weights=weights)
        self.head = MultiLabelHead(self.convnext.classifier[2].in_features, self.n_category, self.n_brand)
        self.convnext.classifier[2] = self.head

        self.accuracy_cat = Accuracy(task="multiclass", num_classes=self.n_category)
        self.accuracy_brand = Accuracy(task="multiclass", num_classes=self.n_brand)
        self.f1_cat = F1Score(task="multiclass", num_classes=self.n_category, average="macro")
        self.f1_brand = F1Score(task="multiclass", num_classes=self.n_brand, average="macro")
        self.criterion_cat = nn.CrossEntropyLoss()
        self.criterion_brand = nn.CrossEntropyLoss()

        self.train_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.val_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.test_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}

    def forward(self, pixel_values):
        cat_logits, brand_logits = self.convnext(pixel_values)

        return cat_logits, brand_logits
    
    def _common_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)
        loss_cat = self.criterion_cat(cat_pred, category)
        loss_brand = self.criterion_cat(brand_pred, brand)
        loss = loss_cat + loss_brand

        return loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand
    
    def training_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cat_loss', loss_cat)
        self.log('train_brand_loss', loss_brand)

        self.train_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.train_output["cat_actual"].append(category)
        self.train_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.train_output["brand_actual"].append(brand)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.val_output["cat_actual"].append(category)
        self.val_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.val_output["brand_actual"].append(brand)

        return self.val_output
    
    def test_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        self.test_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.test_output["cat_actual"].append(category)
        self.test_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.test_output["brand_actual"].append(brand)
        return self.test_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer
                ),
                "monitor": "val_loss",
            }
        }
    
    def predict_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)

        return cat_pred, brand_pred
    
    def on_train_epoch_end(self):
        cat_pred = torch.cat(self.train_output["cat_pred"])
        cat_actual = torch.cat(self.train_output["cat_actual"])
        brand_pred = torch.cat(self.train_output["brand_pred"])
        brand_actual = torch.cat(self.train_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        self.log("train_cat_f1", cat_f1_score)
        self.log("train_cat_accuracy", cat_accuracy_score)
        self.log("train_brand_f1", brand_f1_score)
        self.log("train_brand_accuracy", brand_accuracy_score)
        self.log("train_exact_match_ratio", exact_match_ratio)

        self.train_output["cat_pred"].clear()
        self.train_output["cat_actual"].clear()
        self.train_output["brand_pred"].clear()
        self.train_output["brand_actual"].clear()
    
    def on_validation_epoch_end(self):
        cat_pred = torch.cat(self.val_output["cat_pred"])
        cat_actual = torch.cat(self.val_output["cat_actual"])
        brand_pred = torch.cat(self.val_output["brand_pred"])
        brand_actual = torch.cat(self.val_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        print(f'\nCategory f1_score: {cat_f1_score} | Category accuracy: {cat_accuracy_score}')
        print(f'Brand f1_score: {brand_f1_score} | Brand accuracy: {brand_accuracy_score}')
        print(f'Exact match ratio: {exact_match_ratio}')

        self.log("val_cat_f1", cat_f1_score)
        self.log("val_cat_accuracy", cat_accuracy_score)
        self.log("val_brand_f1", brand_f1_score)
        self.log("val_brand_accuracy", brand_accuracy_score)
        self.log("val_exact_match_ratio", exact_match_ratio)

        self.val_output["cat_pred"].clear()
        self.val_output["cat_actual"].clear()
        self.val_output["brand_pred"].clear()
        self.val_output["brand_actual"].clear()

    def on_test_epoch_end(self):
        cat_pred = torch.cat(self.test_output["cat_pred"])
        cat_actual = torch.cat(self.test_output["cat_actual"])
        brand_pred = torch.cat(self.test_output["brand_pred"])
        brand_actual = torch.cat(self.test_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)

        cat_pred = cat_pred.tolist()
        cat_actual = cat_actual.tolist()
        brand_pred = brand_pred.tolist()
        brand_actual = brand_actual.tolist()

        exact_match_ratio = np.mean((np.array(cat_pred) == np.array(cat_actual)) & (np.array(brand_pred) == np.array(brand_actual)))
        self.log("test_cat_f1", cat_f1_score)
        self.log("test_cat_accuracy", cat_accuracy_score)
        self.log("test_brand_f1", brand_f1_score)
        self.log("test_brand_accuracy", brand_accuracy_score)
        self.log("test_exact_match_ratio", exact_match_ratio)

        print("Exact match ratio :", exact_match_ratio)
        
        print('\nCategory Metrics')
        print(classification_report(cat_actual, cat_pred))
        cm = confusion_matrix(cat_actual, cat_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()
        print('\nBrand Metrics')
        print(classification_report(brand_actual, brand_pred))
        cm = confusion_matrix(brand_actual, brand_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()

        self.test_output["cat_pred"].clear()
        self.test_output["cat_actual"].clear()
        self.test_output["brand_pred"].clear()
        self.test_output["brand_actual"].clear()

class MAXVIT(L.LightningModule):
    def __init__(self, n_category, n_brand):
        super(MAXVIT, self).__init__()
        self.save_hyperparameters()
        self.n_category = n_category
        self.n_brand = n_brand

        self.maxvit = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
        self.head = MultiLabelHead(self.maxvit.classifier[-1].in_features, self.n_category, self.n_brand)
        self.maxvit.classifier[-1] = self.head

        self.accuracy_cat = Accuracy(task="multiclass", num_classes=self.n_category)
        self.accuracy_brand = Accuracy(task="multiclass", num_classes=self.n_brand)
        self.f1_cat = F1Score(task="multiclass", num_classes=self.n_category, average="macro")
        self.f1_brand = F1Score(task="multiclass", num_classes=self.n_brand, average="macro")
        self.criterion_cat = nn.CrossEntropyLoss()
        self.criterion_brand = nn.CrossEntropyLoss()

        self.train_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.val_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.test_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}

    def forward(self, pixel_values):
        cat_logits, brand_logits = self.maxvit(pixel_values)

        return cat_logits, brand_logits
    
    def _common_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)
        loss_cat = self.criterion_cat(cat_pred, category)
        loss_brand = self.criterion_cat(brand_pred, brand)
        loss = loss_cat + loss_brand

        return loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand
    
    def training_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cat_loss', loss_cat)
        self.log('train_brand_loss', loss_brand)

        self.train_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.train_output["cat_actual"].append(category)
        self.train_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.train_output["brand_actual"].append(brand)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.val_output["cat_actual"].append(category)
        self.val_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.val_output["brand_actual"].append(brand)

        return self.val_output
    
    def test_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        self.test_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.test_output["cat_actual"].append(category)
        self.test_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.test_output["brand_actual"].append(brand)
        return self.test_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer
                ),
                "monitor": "val_loss",
            }
        }
    
    def predict_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)

        return cat_pred, brand_pred
    
    def on_train_epoch_end(self):
        cat_pred = torch.cat(self.train_output["cat_pred"])
        cat_actual = torch.cat(self.train_output["cat_actual"])
        brand_pred = torch.cat(self.train_output["brand_pred"])
        brand_actual = torch.cat(self.train_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        self.log("train_cat_f1", cat_f1_score)
        self.log("train_cat_accuracy", cat_accuracy_score)
        self.log("train_brand_f1", brand_f1_score)
        self.log("train_brand_accuracy", brand_accuracy_score)
        self.log("train_exact_match_ratio", exact_match_ratio)

        self.train_output["cat_pred"].clear()
        self.train_output["cat_actual"].clear()
        self.train_output["brand_pred"].clear()
        self.train_output["brand_actual"].clear()
    
    def on_validation_epoch_end(self):
        cat_pred = torch.cat(self.val_output["cat_pred"])
        cat_actual = torch.cat(self.val_output["cat_actual"])
        brand_pred = torch.cat(self.val_output["brand_pred"])
        brand_actual = torch.cat(self.val_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        print(f'\nCategory f1_score: {cat_f1_score} | Category accuracy: {cat_accuracy_score}')
        print(f'Brand f1_score: {brand_f1_score} | Brand accuracy: {brand_accuracy_score}')
        print(f'Exact match ratio: {exact_match_ratio}')

        self.log("val_cat_f1", cat_f1_score)
        self.log("val_cat_accuracy", cat_accuracy_score)
        self.log("val_brand_f1", brand_f1_score)
        self.log("val_brand_accuracy", brand_accuracy_score)
        self.log("val_exact_match_ratio", exact_match_ratio)

        self.val_output["cat_pred"].clear()
        self.val_output["cat_actual"].clear()
        self.val_output["brand_pred"].clear()
        self.val_output["brand_actual"].clear()

    def on_test_epoch_end(self):
        cat_pred = torch.cat(self.test_output["cat_pred"])
        cat_actual = torch.cat(self.test_output["cat_actual"])
        brand_pred = torch.cat(self.test_output["brand_pred"])
        brand_actual = torch.cat(self.test_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)

        cat_pred = cat_pred.tolist()
        cat_actual = cat_actual.tolist()
        brand_pred = brand_pred.tolist()
        brand_actual = brand_actual.tolist()

        exact_match_ratio = np.mean((np.array(cat_pred) == np.array(cat_actual)) & (np.array(brand_pred) == np.array(brand_actual)))
        self.log("test_cat_f1", cat_f1_score)
        self.log("test_cat_accuracy", cat_accuracy_score)
        self.log("test_brand_f1", brand_f1_score)
        self.log("test_brand_accuracy", brand_accuracy_score)
        self.log("test_exact_match_ratio", exact_match_ratio)

        print("Exact match ratio :", exact_match_ratio)
        
        print('\nCategory Metrics')
        print(classification_report(cat_actual, cat_pred))
        cm = confusion_matrix(cat_actual, cat_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()
        print('\nBrand Metrics')
        print(classification_report(brand_actual, brand_pred))
        cm = confusion_matrix(brand_actual, brand_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()

        self.test_output["cat_pred"].clear()
        self.test_output["cat_actual"].clear()
        self.test_output["brand_pred"].clear()
        self.test_output["brand_actual"].clear()

class DenseNet(L.LightningModule):
    def __init__(self, n_category, n_brand):
        super(DenseNet, self).__init__()
        self.save_hyperparameters()
        self.n_category = n_category
        self.n_brand = n_brand

        self.densenet = models.densenet201(weights=models.DenseNet201_Weights)
        self.head = MultiLabelHead(self.densenet.classifier.in_features, self.n_category, self.n_brand)
        self.densenet.classifier = self.head
        self.accuracy_cat = Accuracy(task="multiclass", num_classes=self.n_category)
        self.accuracy_brand = Accuracy(task="multiclass", num_classes=self.n_brand)
        self.f1_cat = F1Score(task="multiclass", num_classes=self.n_category, average="macro")
        self.f1_brand = F1Score(task="multiclass", num_classes=self.n_brand, average="macro")
        self.criterion_cat = nn.CrossEntropyLoss()
        self.criterion_brand = nn.CrossEntropyLoss()

        self.train_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.val_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}
        self.test_output = {"cat_pred": [], "cat_actual": [], "brand_pred": [], "brand_actual": []}

    def forward(self, pixel_values):
        cat_logits, brand_logits = self.densenet(pixel_values)

        return cat_logits, brand_logits
    
    def _common_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)
        loss_cat = self.criterion_cat(cat_pred, category)
        loss_brand = self.criterion_cat(brand_pred, brand)
        loss = loss_cat + loss_brand

        return loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand
    
    def training_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cat_loss', loss_cat)
        self.log('train_brand_loss', loss_brand)

        self.train_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.train_output["cat_actual"].append(category)
        self.train_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.train_output["brand_actual"].append(brand)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.val_output["cat_actual"].append(category)
        self.val_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.val_output["brand_actual"].append(brand)

        return self.val_output
    
    def test_step(self, batch, batch_idx):
        loss, loss_cat, loss_brand, cat_pred, brand_pred, category, brand = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)

        self.test_output["cat_pred"].append(cat_pred.argmax(dim=-1))
        self.test_output["cat_actual"].append(category)
        self.test_output["brand_pred"].append(brand_pred.argmax(dim=-1))
        self.test_output["brand_actual"].append(brand)
        return self.test_output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer
                ),
                "monitor": "val_loss",
            }
        }
    
    def predict_step(self, batch, batch_idx):
        img, category, brand = batch
        cat_pred, brand_pred = self.forward(img)

        return cat_pred, brand_pred
    
    def on_train_epoch_end(self):
        cat_pred = torch.cat(self.train_output["cat_pred"])
        cat_actual = torch.cat(self.train_output["cat_actual"])
        brand_pred = torch.cat(self.train_output["brand_pred"])
        brand_actual = torch.cat(self.train_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        self.log("train_cat_f1", cat_f1_score)
        self.log("train_cat_accuracy", cat_accuracy_score)
        self.log("train_brand_f1", brand_f1_score)
        self.log("train_brand_accuracy", brand_accuracy_score)
        self.log("train_exact_match_ratio", exact_match_ratio)

        self.train_output["cat_pred"].clear()
        self.train_output["cat_actual"].clear()
        self.train_output["brand_pred"].clear()
        self.train_output["brand_actual"].clear()
    
    def on_validation_epoch_end(self):
        cat_pred = torch.cat(self.val_output["cat_pred"])
        cat_actual = torch.cat(self.val_output["cat_actual"])
        brand_pred = torch.cat(self.val_output["brand_pred"])
        brand_actual = torch.cat(self.val_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)
        exact_match_ratio = np.mean((cat_pred.cpu().numpy() == cat_actual.cpu().numpy()) & (brand_pred.cpu().numpy() == brand_actual.cpu().numpy()))

        print(f'\nCategory f1_score: {cat_f1_score} | Category accuracy: {cat_accuracy_score}')
        print(f'Brand f1_score: {brand_f1_score} | Brand accuracy: {brand_accuracy_score}')
        print(f'Exact match ratio: {exact_match_ratio}')

        self.log("val_cat_f1", cat_f1_score)
        self.log("val_cat_accuracy", cat_accuracy_score)
        self.log("val_brand_f1", brand_f1_score)
        self.log("val_brand_accuracy", brand_accuracy_score)
        self.log("val_exact_match_ratio", exact_match_ratio)

        self.val_output["cat_pred"].clear()
        self.val_output["cat_actual"].clear()
        self.val_output["brand_pred"].clear()
        self.val_output["brand_actual"].clear()

    def on_test_epoch_end(self):
        cat_pred = torch.cat(self.test_output["cat_pred"])
        cat_actual = torch.cat(self.test_output["cat_actual"])
        brand_pred = torch.cat(self.test_output["brand_pred"])
        brand_actual = torch.cat(self.test_output["brand_actual"])

        cat_f1_score = self.f1_cat(cat_pred, cat_actual)
        cat_accuracy_score = self.accuracy_cat(cat_pred, cat_actual)
        brand_f1_score = self.f1_brand(brand_pred, brand_actual)
        brand_accuracy_score = self.accuracy_brand(brand_pred, brand_actual)

        cat_pred = cat_pred.tolist()
        cat_actual = cat_actual.tolist()
        brand_pred = brand_pred.tolist()
        brand_actual = brand_actual.tolist()

        exact_match_ratio = np.mean((np.array(cat_pred) == np.array(cat_actual)) & (np.array(brand_pred) == np.array(brand_actual)))
        self.log("test_cat_f1", cat_f1_score)
        self.log("test_cat_accuracy", cat_accuracy_score)
        self.log("test_brand_f1", brand_f1_score)
        self.log("test_brand_accuracy", brand_accuracy_score)
        self.log("test_exact_match_ratio", exact_match_ratio)

        print("Exact match ratio :", exact_match_ratio)
        
        print('\nCategory Metrics')
        print(classification_report(cat_actual, cat_pred))
        cm = confusion_matrix(cat_actual, cat_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()
        print('\nBrand Metrics')
        print(classification_report(brand_actual, brand_pred))
        cm = confusion_matrix(brand_actual, brand_pred)
        dis = ConfusionMatrixDisplay(cm)
        
        dis.plot()
        plt.show()

        self.test_output["cat_pred"].clear()
        self.test_output["cat_actual"].clear()
        self.test_output["brand_pred"].clear()
        self.test_output["brand_actual"].clear()