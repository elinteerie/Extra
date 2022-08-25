import torch


def make_train_step(model, loss_fn, optimizer):
  '''
  Builds The Training step that performs the train loop
  '''
  def perform_train_step(x, y):
    #Set the model to train
    model.train()
    
    #Computes Prediction
    y_pred = model(x)
    #xCompute the loss
    loss = lossfn(y_pred, y)

    loss.backward
    optimizer.step
    optimizer.zero_grad()
    return loss.item()
  return perform_train_step
