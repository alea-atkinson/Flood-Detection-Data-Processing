#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, importlib.util, math, random
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

METHODS = {"denoising", "weak_simsiam", "hybrid_mae_simsiam"}

def seed_everything(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def import_fine_tune(repo_root:Path)->Any:
    path = repo_root / "training" / "fine_tune.py"
    spec = importlib.util.spec_from_file_location("fine_tune_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def read_uavsar_csv(csv_path:Path)->list[Path]:
    rows=[]
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader=csv.DictReader(f)
        if "uavsar_path" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} missing uavsar_path")
        for row in reader:
            rows.append(Path(row["uavsar_path"]))
    if not rows: raise ValueError(f"No rows in {csv_path}")
    return rows

def resolve(path:Path, repo_root:Path)->Path:
    return path if path.is_absolute() else repo_root / path

def load_uavsar(path:Path)->tuple[torch.Tensor, torch.Tensor]:
    with rasterio.open(path) as src:
        sar=src.read().astype(np.float32)
    if sar.shape[0] < 3:
        raise ValueError(f"Expected at least 3 bands in {path}")
    sar=sar[:3]
    valid=~(sar==0).all(axis=0)
    out=sar.copy()
    for c in range(3):
        band=out[c]
        vals=band[valid]
        if vals.size == 0:
            band[:]=0.0; out[c]=band; continue
        lo,hi=np.percentile(vals,[1.0,99.0])
        vals=np.clip(vals,lo,hi)
        mean=vals.mean(); std=vals.std()
        if std < 1e-6: band[:]=0.0
        else: band[valid]=(vals-mean)/std
        band[~valid]=0.0
        out[c]=band
    return torch.from_numpy(out.astype(np.float32)), torch.from_numpy(valid.astype(np.float32))[None]

def mild_corruption(x:torch.Tensor, valid:torch.Tensor)->torch.Tensor:
    y=x.clone()
    if random.random()<0.9: y=y*random.uniform(0.80,1.20)
    if random.random()<0.9: y=y+torch.randn_like(y)*random.uniform(0.03,0.10)
    if random.random()<0.5: y=y*(1.0+torch.randn_like(y)*random.uniform(0.03,0.08))
    if random.random()<0.35: y=F.avg_pool2d(y.unsqueeze(0),3,1,1).squeeze(0)
    return (y*valid).contiguous()

def weak_aug(x:torch.Tensor, valid:torch.Tensor)->torch.Tensor:
    y=x.clone(); v=valid
    if random.random()<0.5:
        y=torch.flip(y,dims=[2]); v=torch.flip(v,dims=[2])
    if random.random()<0.8: y=y*random.uniform(0.90,1.10)
    if random.random()<0.5: y=y+torch.randn_like(y)*random.uniform(0.01,0.04)
    if random.random()<0.15: y=F.avg_pool2d(y.unsqueeze(0),3,1,1).squeeze(0)
    return (y*v).contiguous()

def patch_mask(valid_np:np.ndarray, patch_size:int, mask_ratio:float, min_valid_frac:float)->np.ndarray:
    h,w=valid_np.shape
    h=(h//patch_size)*patch_size; w=(w//patch_size)*patch_size
    valid_np=valid_np[:h,:w]
    nr,nc=h//patch_size,w//patch_size
    pm=np.zeros((nr,nc),dtype=bool)
    eligible=[]
    for r in range(nr):
        for c in range(nc):
            block=valid_np[r*patch_size:(r+1)*patch_size,c*patch_size:(c+1)*patch_size]
            if block.mean() >= min_valid_frac:
                eligible.append((r,c))
    random.shuffle(eligible)
    for r,c in eligible[:int(len(eligible)*mask_ratio)]:
        pm[r,c]=True
    return np.kron(pm, np.ones((patch_size,patch_size), dtype=bool))

class SSLDataset(Dataset):
    def __init__(self, paths:list[Path], repo_root:Path, method:str, patch_size:int, mask_ratio:float, min_valid_frac:float):
        self.paths=[resolve(p,repo_root) for p in paths]
        self.method=method; self.patch_size=patch_size; self.mask_ratio=mask_ratio; self.min_valid_frac=min_valid_frac
    def __len__(self): return len(self.paths)
    def __getitem__(self, i:int):
        clean,valid=load_uavsar(self.paths[i])
        _,h,w=clean.shape
        h=(h//self.patch_size)*self.patch_size; w=(w//self.patch_size)*self.patch_size
        clean=clean[:,:h,:w]; valid=valid[:,:h,:w]
        if self.method=="denoising":
            return mild_corruption(clean,valid), clean, valid
        if self.method=="weak_simsiam":
            return weak_aug(clean,valid), weak_aug(clean,valid)
        if self.method=="hybrid_mae_simsiam":
            m=patch_mask(valid[0].numpy().astype(bool), self.patch_size, self.mask_ratio, self.min_valid_frac)
            mt=torch.from_numpy(m.astype(np.float32))[None]
            loss_mask=mt*valid
            masked=clean.clone(); masked[:,m]=0.0
            return masked, clean, loss_mask, weak_aug(masked,valid), weak_aug(masked,valid)
        raise ValueError(self.method)

class Encoder(nn.Module):
    def __init__(self, unet:nn.Module):
        super().__init__(); self.unet=unet
        missing=[n for n in ["enc1","enc2","enc3","enc4","bottleneck","pool"] if not hasattr(unet,n)]
        if missing: raise AttributeError(f"UNet missing {missing}")
    def forward(self,x):
        e1=self.unet.enc1(x); e2=self.unet.enc2(self.unet.pool(e1)); e3=self.unet.enc3(self.unet.pool(e2)); e4=self.unet.enc4(self.unet.pool(e3))
        return self.unet.bottleneck(self.unet.pool(e4))

class SimHead(nn.Module):
    def __init__(self, unet:nn.Module, image_size:int, proj:int, hidden:int):
        super().__init__(); self.encoder=Encoder(unet); self.pool=nn.AdaptiveAvgPool2d((1,1))
        with torch.no_grad():
            dim=self.pool(self.encoder(torch.zeros(1,3,image_size,image_size))).flatten(1).shape[1]
        self.projector=nn.Sequential(nn.Linear(dim,hidden),nn.BatchNorm1d(hidden),nn.ReLU(inplace=True),
                                     nn.Linear(hidden,hidden),nn.BatchNorm1d(hidden),nn.ReLU(inplace=True),
                                     nn.Linear(hidden,proj),nn.BatchNorm1d(proj,affine=False))
        self.predictor=nn.Sequential(nn.Linear(proj,hidden//2),nn.BatchNorm1d(hidden//2),nn.ReLU(inplace=True),
                                     nn.Linear(hidden//2,proj))
    def one(self,x):
        z=self.projector(self.pool(self.encoder(x)).flatten(1)); p=self.predictor(z); return p,z.detach()
    def forward(self,x1,x2):
        p1,z1=self.one(x1); p2,z2=self.one(x2); return p1,p2,z1,z2

class SSLModel(nn.Module):
    def __init__(self, unet, method, image_size, proj, hidden):
        super().__init__(); self.unet=unet; self.method=method
        self.sim = SimHead(unet,image_size,proj,hidden) if method in {"weak_simsiam","hybrid_mae_simsiam"} else None

def masked_mse(pred,target,mask,eps=1e-6):
    return (((pred-target)**2)*mask).sum()/(mask.sum()*pred.shape[1]+eps)

def negcos(p,z):
    p=F.normalize(p,dim=1); z=F.normalize(z,dim=1); return -(p*z).sum(dim=1).mean()

def simloss(p1,p2,z1,z2):
    return 0.5*(negcos(p1,z2)+negcos(p2,z1))

def run_epoch(model, loader, device, optimizer, method, lam):
    train=optimizer is not None; model.train(train)
    tl=tr=ts=0.0; n=0
    for batch in loader:
        if train: optimizer.zero_grad(set_to_none=True)
        if method=="denoising":
            corrupted,clean,valid=[b.to(device,non_blocking=True) for b in batch]
            pred=model.unet(corrupted); r=masked_mse(pred,clean,valid); s=torch.zeros((),device=device); loss=r
        elif method=="weak_simsiam":
            v1,v2=[b.to(device,non_blocking=True) for b in batch]
            p1,p2,z1,z2=model.sim(v1,v2); s=simloss(p1,p2,z1,z2); r=torch.zeros((),device=device); loss=s
        elif method=="hybrid_mae_simsiam":
            masked,clean,lmask,v1,v2=[b.to(device,non_blocking=True) for b in batch]
            pred=model.unet(masked); r=masked_mse(pred,clean,lmask)
            p1,p2,z1,z2=model.sim(v1,v2); s=simloss(p1,p2,z1,z2); loss=r+lam*s
        else: raise ValueError(method)
        if train:
            loss.backward(); optimizer.step()
        tl+=float(loss.item()); tr+=float(r.item()); ts+=float(s.item()); n+=1
    if n==0: raise ValueError("No batches")
    return tl/n,tr/n,ts/n

def transferable(unet):
    return {k:v for k,v in unet.state_dict().items() if not k.startswith("out.")}

def safe_args(args):
    return {k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--method", choices=sorted(METHODS), required=True)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--train-csv", type=Path, default=Path("training/pretrain_csvs/train.csv"))
    p.add_argument("--val-csv", type=Path, default=Path("training/pretrain_csvs/val.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("training/pretrain_weights"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--min-valid-frac", type=float, default=0.9)
    p.add_argument("--projection-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lambda-simsiam", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=98)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    return p.parse_args()

def main():
    args=parse_args(); seed_everything(args.seed)
    repo=args.repo_root.resolve(); out_dir=(repo/args.out_dir).resolve(); res_dir=(repo/args.results_dir).resolve()
    out_dir.mkdir(parents=True,exist_ok=True); res_dir.mkdir(parents=True,exist_ok=True)
    fine_tune=import_fine_tune(repo)
    train=read_uavsar_csv((repo/args.train_csv).resolve()); val=read_uavsar_csv((repo/args.val_csv).resolve())
    if args.max_train_samples is not None: train=train[:args.max_train_samples]
    if args.max_val_samples is not None: val=val[:args.max_val_samples]
    if len(train)<args.batch_size or len(val)<args.batch_size: raise ValueError("Not enough samples for batch size")
    device=torch.device(args.device); pin=device.type=="cuda"
    train_loader=DataLoader(SSLDataset(train,repo,args.method,args.patch_size,args.mask_ratio,args.min_valid_frac), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader=DataLoader(SSLDataset(val,repo,args.method,args.patch_size,args.mask_ratio,args.min_valid_frac), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    out_ch=1 if args.method=="weak_simsiam" else 3
    unet=fine_tune.UNet(in_channels=3,out_channels=out_ch,base_channels=args.base_channels)
    model=SSLModel(unet,args.method,args.image_size,args.projection_dim,args.hidden_dim).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt=out_dir/f"best_{args.run_name}.pth"; metrics=res_dir/f"{args.run_name}_pretrain_metrics.csv"
    with metrics.open("w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_recon_loss","train_simsiam_loss","val_loss","val_recon_loss","val_simsiam_loss","best_val_loss"])
    print(f"Using device: {device}")
    print(f"Method: {args.method}")
    print(f"Train samples: {len(train)}")
    print(f"Val samples: {len(val)}")
    print(f"Checkpoint path: {ckpt}")
    print(f"Metrics path: {metrics}")
    best=math.inf
    for epoch in range(1,args.epochs+1):
        train_loss,train_rec,train_sim=run_epoch(model,train_loader,device,opt,args.method,args.lambda_simsiam)
        with torch.no_grad():
            val_loss,val_rec,val_sim=run_epoch(model,val_loader,device,None,args.method,args.lambda_simsiam)
        saved=""
        if val_loss<best:
            best=val_loss
            torch.save({"model_state_dict":transferable(model.unet),"ssl_variant_state_dict":model.state_dict(),
                        "optimizer_state_dict":opt.state_dict(),"epoch":epoch,"val_loss":val_loss,
                        "best_val_loss":best,"pretrain_method":args.method,"args":safe_args(args)}, ckpt)
            saved="saved"
        with metrics.open("a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([epoch,train_loss,train_rec,train_sim,val_loss,val_rec,val_sim,best])
        print(f"Epoch {epoch:03d}/{args.epochs:03d} train_loss={train_loss:.5f} train_recon={train_rec:.5f} train_sim={train_sim:.5f} val_loss={val_loss:.5f} val_recon={val_rec:.5f} val_sim={val_sim:.5f} best_val_loss={best:.5f} {saved}")
    print(f"Done. Best checkpoint: {ckpt}")
    print(f"Metrics CSV: {metrics}")

if __name__=="__main__":
    main()
