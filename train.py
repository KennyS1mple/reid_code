# CACIOUS CODING
# Data     : 6/1/23  7:51 PM
# File name: train
# Desc     :script for training reid

from opt import opts
from model.model import *
from torch.utils.data import DataLoader
from dataset.my_dataset import MyDataset
from torch import nn
from network.reid_training_model import TrainingModel

# 获取参数
opt = opts()

opt.device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                          if opt.gpus[0] >= 0 else 'cpu')
device = opt.device
my_dataset = MyDataset(opt)
model = create_model(opt.arch, opt.heads, opt.head_conv)
training_model = TrainingModel(model, opt).to(device)

dataloader = DataLoader(my_dataset, batch_size=opt.batch_size, shuffle=True)
print("REID MODEL's parameters: " + str(opt))
print("running on gpu." if torch.cuda.is_available() else "running on cpu.")
loss_func = nn.CrossEntropyLoss()

training_model.train()
optimizer = torch.optim.Adam(training_model.parameters(), opt.lr)

print("")
print("start training...")
for epoch in range(opt.epochs):
    correct = 0
    for data_batch, label_batch in dataloader:
        data_batch = data_batch.to(opt.device)
        label_batch = label_batch.to(opt.device)
        output = training_model(data_batch)
        loss = loss_func(output, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, preds = torch.max(output, 1)
        correct += torch.sum(preds == label_batch)
    # scheduler.step()
    print("epoch: " + str(epoch) + "---loss: " + str(loss.item()) +
          "---acc: " + str((correct / len(my_dataset)).item()))

    if epoch % 10 == 0 and epoch > 0:
        save_model("./weight/dla_training_weights_{}.pth".format(epoch),
                   epoch, training_model, optimizer)
        print(f"model saved successfully.epoch:{epoch}")
    if opt.epochs - epoch < 10:
        save_model("./weight/dla_training_weights_{}.pth".format(epoch),
                   epoch, training_model, optimizer)
        print(f"model saved successfully.epoch:{epoch}")

