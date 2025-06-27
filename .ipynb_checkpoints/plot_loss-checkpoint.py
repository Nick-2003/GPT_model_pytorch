import matplotlib.pyplot as plt


training_stats = open('nohup.out', 'r', encoding='utf-8')
train_loss = []
for i in training_stats:
    if 'Training Loss' in i:
        try:
            y = i.split()
            train_loss.append(float(y[-1].strip()))
        except IndexError:
            continue
epochs = [i for i in range(1, len(train_loss)+1)]
plt.plot(epochs, train_loss, '-m', label='Train Loss')
plt.title('Train Loss vs Epoch')
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss.png')
plt.clf()

