from tqdm import tqdm

def train_model(model,trainloader,criterion,optimizer,device = "cuda"):
    model.train()
    total_loss = 0
    
    for anchor,positive,negative in tqdm(trainloader,desc = "Train Batches"):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        anchor = model(anchor)
        positive = model(positive)
        negative = model(negative)
        loss = criterion(anchor,positive,negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(trainloader)

