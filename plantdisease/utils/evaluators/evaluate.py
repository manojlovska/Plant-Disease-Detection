import torch
import torch.nn.functional as F
from loguru import logger

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch, device):
    images, labels = batch
    images.to(device)
    labels.to(device)

    out = model(images)
    out.to(device)                   # Generate prediction
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)          # Calculate accuracy

    return {"val_loss": loss.detach(), "val_accuracy": acc}


@torch.no_grad()
def evaluate(model, val_loader, device, return_predictions = False):
    model.eval()
    outputs = [validation_step(model,batch, device) for batch in val_loader]

    batch_losses = [x["val_loss"] for x in outputs]
    batch_accuracy = [x["val_accuracy"] for x in outputs]

    epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
    epoch_accuracy = torch.stack(batch_accuracy).mean()

    # if return_predictions:
    #     return predictions, {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    # else:

    return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}
