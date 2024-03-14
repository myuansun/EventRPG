import torch

def TET_loss(outputs, labels, criterion=torch.nn.CrossEntropyLoss(), means=1., lamb=0.001, mode="TB"):
    if mode == "TB":
        T = outputs.size(0) # T, B, C, H, W
    else:
        T = outputs.size(1) # B, T, C, H, W
    Loss_es = 0
    for t in range(T):
        if mode == "TB":
            Loss_es += criterion(outputs[t, :, ...], labels)
        else:
            Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd