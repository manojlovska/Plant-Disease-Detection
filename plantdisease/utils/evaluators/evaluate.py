import torch
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch):
    images, labels = batch
    out = model(images)                   # Generate prediction
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)          # Calculate accuracy

    return out, {"val_loss": loss.detach(), "val_accuracy": acc}


@torch.no_grad()
def evaluate(model, val_loader, return_predictions = False):
    model.eval()
    predictions, outputs = [validation_step(batch) for batch in val_loader]

    batch_losses = [x["val_loss"] for x in outputs]
    batch_accuracy = [x["val_accuracy"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
    epoch_accuracy = torch.stack(batch_accuracy).mean()

    if return_predictions:
        return predictions, {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    else:
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}
