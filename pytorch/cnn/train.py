import torch

def mnist_train(model, num_epochs, train_dl, validation_dl, loss_fn, optimizer):
    training_loss_hist = [0] * num_epochs
    training_accuracy_hist = [0] * num_epochs
    validation_loss_hist = [0] * num_epochs
    validation_accuracy_hist = [0] * num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss_hist[epoch] = training_loss_hist[epoch] + loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            training_accuracy_hist[epoch] = training_accuracy_hist[epoch] + is_correct.sum()

        training_loss_hist[epoch] = training_loss_hist[epoch] / len(train_dl.dataset)
        training_accuracy_hist[epoch] = training_accuracy_hist[epoch] / len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                validation_loss_hist[epoch] = validation_loss_hist[epoch] + loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                validation_accuracy_hist[epoch] = validation_accuracy_hist[epoch] + is_correct.sum()


        validation_loss_hist[epoch] = validation_loss_hist[epoch] / len(validation_dl.dataset)
        validation_accuracy_hist[epoch] = validation_accuracy_hist[epoch] / len(validation_dl.dataset)
            

        print(f'Epoch {epoch+1} | Training Accuracy: {training_accuracy_hist[epoch]} | Validation Accuracy: {validation_accuracy_hist[epoch]}')


def celeba_train(model, num_epochs, train_dl, validation_dl, loss_fn, optimizer):
    training_loss_hist = [0] * num_epochs
    training_accuracy_hist = [0] * num_epochs
    validation_loss_hist = [0] * num_epochs
    validation_accuracy_hist = [0] * num_epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'device = {device}')
    
    model = model.cuda()
    
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            pred = model(x_batch)[:,0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss_hist[epoch] = training_loss_hist[epoch] + loss.item()*y_batch.size(0)
            is_correct = ((pred>=0.5).float() == y_batch).float()
            training_accuracy_hist[epoch] = training_accuracy_hist[epoch] + is_correct.sum()

        training_loss_hist[epoch] = training_loss_hist[epoch] / len(train_dl.dataset)
        training_accuracy_hist[epoch] = training_accuracy_hist[epoch] / len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_dl:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                pred = model(x_batch)[:,0]
                loss = loss_fn(pred, y_batch.float())
                validation_loss_hist[epoch] = validation_loss_hist[epoch] + loss.item()*y_batch.size(0)
                is_correct = ((pred>=0.5) == y_batch).float()
                validation_accuracy_hist[epoch] = validation_accuracy_hist[epoch] + is_correct.sum()


        validation_loss_hist[epoch] = validation_loss_hist[epoch] / len(validation_dl.dataset)
        validation_accuracy_hist[epoch] = validation_accuracy_hist[epoch] / len(validation_dl.dataset)
            

        print(f'Epoch {epoch+1} | Training Accuracy: {training_accuracy_hist[epoch]} | Validation Accuracy: {validation_accuracy_hist[epoch]}')

        
