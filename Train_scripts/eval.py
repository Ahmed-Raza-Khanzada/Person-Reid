import torch
from tqdm import tqdm
def eval_model(model,trainloader,criterion,device = "cuda"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for anchor,positive,negative in tqdm(trainloader,desc = "Eval Batches"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor = model(anchor)
            positive = model(positive)
            negative = model(negative)
            loss = criterion(anchor,positive,negative)
            total_loss += loss.item()
    return total_loss/len(trainloader)
