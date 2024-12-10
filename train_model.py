from dataset import create_loader
from models import ResNet101, VIT, CONVNEXT, MAXVIT, DenseNet
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

train_loader, val_loader, test_loader, cat_enc, brand_enc = create_loader()

n_category = len(cat_enc.classes_)
n_brand = len(brand_enc.classes_)

resnet101 = ResNet101(n_category, n_brand)

resnet101_logger = CSVLogger("logs", name="resnet101")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=30, callbacks=[early_stop_callback], logger=resnet101_logger)
trainer.fit(resnet101, train_loader, val_loader)
trainer.test(resnet101, test_loader)

vit = VIT(n_category, n_brand)

VIT_logger = CSVLogger("logs", name="VIT")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=30, callbacks=[early_stop_callback], logger=VIT_logger)
trainer.fit(vit, train_loader, val_loader)
trainer.test(vit, test_loader)

convnext = CONVNEXT(n_category, n_brand)

convnext_logger = CSVLogger("logs", name="convnext")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=30, callbacks=[early_stop_callback], logger=convnext_logger)
trainer.fit(convnext, train_loader, val_loader)
trainer.test(convnext, test_loader)

maxvit = MAXVIT(n_category, n_brand)

maxvit_logger = CSVLogger("logs", name="maxvit")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=30, callbacks=[early_stop_callback], logger=maxvit_logger)
trainer.fit(maxvit, train_loader, val_loader)
trainer.test(maxvit, test_loader)

densenet = DenseNet(n_category, n_brand)

densenet_logger = CSVLogger("logs", name="densenet")
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=30, callbacks=[early_stop_callback], logger=densenet_logger)
trainer.fit(densenet, train_loader, val_loader)
trainer.test(densenet, test_loader)