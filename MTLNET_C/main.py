import torch
import torch.nn as nn
from data import data_preparation
from mtl_networkC import MTL_NetworkC
from utils import random_mini_batches


LR = 0.006898443729831321
mb_size = 100

def main():
    # Train model
    print()
    print("Training...")
    X_train, y1_train, y2_train, X_test, y1_test, y2_test = data_preparation(num_feature=100, rho=0.8, num_row = 10000)
    num_inputs = X_train.shape[1]
    MTL = MTL_NetworkC(feature_size=num_inputs,hidden_layer_size=54, task_layer1=5, task_layer2=9, n_output_1=1, n_output_2=1, n_output_3=1)
    optimizer = torch.optim.Adam(MTL.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    input_size, feature_size = X_train.shape
    for it in range(1000):
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        epoch_cost3 = 0
        epoch_cost4 = 0
        cost1tr = []
        cost2tr = []
        cost3tr = []
        cost4tr = []
        costtr = []
        num_minibatches = int(input_size / mb_size)
        for minibatch in (random_mini_batches(X_train, y1_train, y2_train, mb_size)):
            X_batch, Y1_batch, Y2_batch  = minibatch
            Y1_pred, Y2_pred, Y3_pred, Y4_pred = MTL(X_batch)
            loss1 = loss_func(Y1_pred, Y1_batch.view(-1,1))
            loss2 = loss_func(Y2_pred, Y2_batch.view(-1,1))
            loss3 = loss_func(Y3_pred, Y1_batch.view(-1,1))
            loss4 = loss_func(Y4_pred, Y2_batch.view(-1,1))
            comb_loss = loss1+ loss2
            loss = comb_loss + loss3 + loss4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (comb_loss/ num_minibatches)
            #epoch_cost2 = epoch_cost2 + (loss2 / num_minibatches)
            epoch_cost3 = epoch_cost3 + (loss3 / num_minibatches)
            epoch_cost4 = epoch_cost4 + (loss4 / num_minibatches)
        costtr.append(torch.mean(epoch_cost))
        cost1tr.append(torch.mean(epoch_cost1))
        #cost2tr.append(torch.mean(epoch_cost2))
        cost3tr.append(torch.mean(epoch_cost3))
        cost4tr.append(torch.mean(epoch_cost4))
        MTL.eval()
        with torch.no_grad():
            val1_predict, val2_predict, val3_predict, val4_predict = MTL(X_test)
            l1D = loss_func(val1_predict, y1_test.view(-1,1))
            l2D = loss_func(val2_predict, y2_test.view(-1,1))
            l3D = loss_func(val3_predict, y1_test.view(-1,1))
            l4D = loss_func(val4_predict, y2_test.view(-1,1))
            comb_val_loss = l1D+l2D
            cost1D = []
            cost2D = []
            cost3D = []
            cost4D = []
            costD = []
            cost1D.append(comb_val_loss)
            #cost2D.append(l2D)
            cost3D.append(l3D)
            cost4D.append(l4D)
            costD.append(comb_val_loss+l3D+l4D)
        print("Epoch: ", it, " Train Loss1: ", cost3tr[-1], " Val Loss1: ", cost3D[-1], " Train Loss2: ", cost4tr[-1], " Val Loss2: ", cost4D[-1])

        #print("Epoch: ", it, " Training Loss: ", costtr[-1], " Validation Loss: ", costD[-1])
                    
if __name__ == '__main__':
    main()