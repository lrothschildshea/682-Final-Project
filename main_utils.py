import torch #pylint: disable= E0401
import torch.nn.functional as F #pylint: disable= E0401

def check_accuracy(loader, model, device, train):
    
    if train:
        print('        Checking accuracy on validation set')
    else:
        print('    Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode

    all_scores = torch.tensor([])
    out = torch.tensor([])
    all_scores = all_scores.to(device = device, dtype = torch.float)
    out = out.to(device=device, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)
            if not train:
                out = torch.cat((out, preds), 0)
                all_scores = torch.cat((all_scores, scores), 0)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples

        if train:
            print('         %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        else:
            print('     %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print()
            return out, all_scores, num_correct

def train_model(model, optimizer, device, loader_train, loader_val, epochs=1):
    model = model.to(device=device)
    for e in range(epochs):
        print('    epoch #' + str(e + 1))
        print()
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 175 == 0:
                print('        Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model, device, True)
                print()

def combine_labels(labelset, scoreset, num_labels, device):
    labels = torch.ones(10000, dtype=torch.long) * 10
    labels = labels.to(device=device, dtype=torch.long)
    print('Combining Labels')
    for i in range(num_labels):
        for j in range(10000):
            if labelset[i][j] == 1:
                if labels[j] == 10:
                    labels[j] = i
                elif  torch.abs(scoreset[labels[j]][j][0] - scoreset[labels[j]][j][1]) < torch.abs(scoreset[i][j][0] - scoreset[i][j][1]):
                    labels[j] = i
    return labels

def count_collisions(labelset, num_labels, device):
    print('Counting label collisions and unlabled images')
    labels = torch.zeros(10000, dtype=torch.long)
    labels = labels.to(device=device, dtype=torch.long)
    collision = 0
    unlabeled = 0

    for i in range(num_labels):
        labels += labelset[i]
    for i in range(10000):
        if labels[i] > 1:
            collision += labels[i]-1
        if labels[i] == 0:
            unlabeled += 1

    print("%d Label Collsions" % collision)
    print("%d unlabeled" % unlabeled)
    print()
