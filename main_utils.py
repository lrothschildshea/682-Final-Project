import torch
import torch.nn.functional as F

def check_accuracy(loader, model, device, dtype, score):
    
    '''if loader.dataset.train:
        print('        Checking accuracy on validation set')
    else:
        print('    Checking accuracy on test set')   '''
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    out = torch.tensor([])
    out = out.to(device=device, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            #y[y == score] = 10
            #y[y != 10] = 0
            #y[y == 10] = 1
            
            scores = model(x)
            _, preds = scores.max(1)
            #if not loader.dataset.train:
            #    out = torch.cat((out, preds), 0)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples

        #temp print statement until we fix the following variable
        print('        Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

        '''if loader.dataset.train:
            print('        Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        else:
            print('    Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print()
            return out'''

def train_model(model, optimizer, device, dtype, loader_train, loader_val, score, epochs=1):

    model = model.to(device=device)
    for e in range(epochs):
        print('    epoch #' + str(e + 1))
        print()
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            
            #y[y == score] = 10
            #y[y != 10] = 0
            #y[y == 10] = 1

            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 100 == 0:
                print('        Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model, device, dtype, score)
                print()
