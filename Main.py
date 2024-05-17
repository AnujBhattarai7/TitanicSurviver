import pandas as pd

df = pd.read_csv("train.csv")

InputColums = ["Pclass", "Sex", "Parch", "SibSp", "Fare"]
SurviveColums = ["Survived"]

Colums2Idx = {y : x for x, y in enumerate(InputColums)}
print(Colums2Idx)

df.dropna(subset=InputColums)
df.dropna(subset=SurviveColums)
Inputdata = df[InputColums].values
Survivedata = df[SurviveColums].values

print(len(Inputdata))
print(len(Survivedata))

Sex2Idx = {'male':0, 'female':1}
print(Sex2Idx)
print(Inputdata[0])

import torch
InputTensor = torch.zeros(size=(len(Inputdata), len(InputColums)), dtype=torch.float)  
TargetTensor = torch.zeros(size=(len(Survivedata), len(SurviveColums)), dtype=torch.float)  

for x in range(len(Inputdata)):
    InputTensor[x][Colums2Idx["Pclass"]] = Inputdata[x][Colums2Idx["Pclass"]]
    InputTensor[x][Colums2Idx["Sex"]] = Sex2Idx[Inputdata[x][Colums2Idx["Sex"]]]
    InputTensor[x][Colums2Idx["Parch"]] = Inputdata[x][Colums2Idx["Parch"]]
    InputTensor[x][Colums2Idx["Fare"]] = Inputdata[x][Colums2Idx["Fare"]]
    TargetTensor[x][0] = Survivedata[x][0]

print(InputTensor[:10])
print(TargetTensor[:10])

# Model
from torch import nn

class TitanicModel(nn.Module):
    def __init__(self, InputSize, HiddenSize, OutputSize, NLayers) -> None:
        super().__init__()

        self.Layers = nn.ModuleList()
        self.CreateLayers(InputSize, HiddenSize, OutputSize, NLayers)

    def CreateLayers(self, InputSize, HiddenSize, OutputSize, NLayers):
        self.Layers.append(nn.Sequential(
                nn.Linear(InputSize, HiddenSize),
                # nn.ReLU()
        ))
        # self.Layers.append(nn.ReLU())
        
        for x in range(NLayers):
            self.Layers.append(nn.Sequential(
                nn.Linear(HiddenSize, HiddenSize*2),
                # nn.ReLU()
            ))
            HiddenSize *= 2
        
        self.Layers.append(nn.Sequential(
            nn.Linear(HiddenSize, OutputSize),
            nn.Sigmoid()
        ))

    def forward(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x
    
LEARNING_RATE = 0.001
HIDDEN_SIZE = 16
NLAYERS = 1
NEPOCHS = 1

Model = TitanicModel(
    InputSize=len(InputColums),
    HiddenSize=HIDDEN_SIZE,
    OutputSize=len(SurviveColums),
    NLayers=NLAYERS
)

print(Model)

Model.load_state_dict(torch.load(f="Model/NonReLUModel.pth"))
# Model.load_state_dict(torch.load(f="Model/Model.pth"))

LossFn = nn.BCELoss()
Optim = torch.optim.SGD(params=Model.parameters(), lr=LEARNING_RATE)

# Accuracy Find
y_acc_pred = torch.round(Model(InputTensor))
Acc = 0
for x in range(len(y_acc_pred)):
    if y_acc_pred[x] == TargetTensor[x]:
        Acc += 1
print(f"Accuracy: {Acc*100/len(y_acc_pred):.2f}%")

for epoch in range(NEPOCHS):
    Model.train()

    y_pred = Model(InputTensor)    
    print(y_pred.shape)

    Loss = LossFn(y_pred, TargetTensor)

    Optim.zero_grad()
    Loss.backward()
    Optim.step()

    print(f"Epoch: {epoch}/{NEPOCHS} Loss: {Loss.item():.4f}")

torch.save(obj=Model.state_dict(), f="Model/NonReLUModel.pth")
# torch.save(obj=Model.state_dict(), f="Model/Model.pth")
