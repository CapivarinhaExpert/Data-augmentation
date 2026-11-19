import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# ---------- 1. Transforms ----------
bbox_train = A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3)
bbox_val = A.BboxParams(format="pascal_voc", label_fields=["labels"])

train_tf = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    ToTensorV2(),
], bbox_params=bbox_train)

val_tf = A.Compose([
    A.Resize(512, 512),
    ToTensorV2(),
], bbox_params=bbox_val)


# ---------- 2. Dataset ----------
class GlomerulosDataset(Dataset):
    def __init__(self, amostras, transform=None):
        # amostras: lista de (caminho_da_imagem, caixas, rotulos)
        # caixas: [[x_min, y_min, x_max, y_max], ...]  rotulos: [1, 1, ...]
        self.amostras = amostras
        self.transform = transform

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, i):
        caminho, caixas, rotulos = self.amostras[i]
        img = cv2.cvtColor(cv2.imread(caminho), cv2.COLOR_BGR2RGB)

        if self.transform:
            out = self.transform(image=img, bboxes=caixas, labels=rotulos)
            img, caixas, rotulos = out["image"], out["bboxes"], out["labels"]

        alvo = {
            "boxes": torch.as_tensor(caixas, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(rotulos, dtype=torch.int64),
        }
        return img.float() / 255.0, alvo


# ---------- 3. Suas anotações ----------
# Aqui você preenche com os seus dados. Exemplo do formato esperado:
amostras_treino = [
    ("dados/treino/img1.png", [[120, 80, 200, 160], [300, 310, 380, 395]], [1, 1]),
    ("dados/treino/img2.png", [[50, 60, 130, 140]], [1]),
]
amostras_val = [
    ("dados/validacao/img3.png", [[200, 220, 290, 300]], [1]),
]


# ---------- 4. Datasets e loaders ----------
train_ds = GlomerulosDataset(amostras_treino, transform=train_tf)
val_ds = GlomerulosDataset(amostras_val, transform=val_tf)

def collate(lote):
    return tuple(zip(*lote))

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)


# ---------- 5. Teste visual ----------
def salvar_exemplos(ds, n=6, prefixo="teste_aug"):
    for k in range(n):
        img, alvo = ds[0]  # sempre a mesma imagem, pra ver o sorteio mudando
        arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8").copy()
        for x1, y1, x2, y2 in alvo["boxes"].int().tolist():
            cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"{prefixo}_{k}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    salvar_exemplos(train_ds)
    print("Imagens salvas. Confere se as caixas continuam em cima dos glomérulos.")