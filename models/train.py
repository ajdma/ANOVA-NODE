import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import copy
from torch import optim
import pickle

from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import roc_auc_score

from .odst import ODST
from models import arch
from models import nn_utils
from models import utils


from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=1000, random_state=0,output_distribution='uniform')


def Trainer(data_x,
            data_y, 
            tree_num , 
            max_order , 
            device, 
            model_path ,
            measure ,
            regression,
            random_state,
            multiclass = 2 ,
            lr_rate =0.01 ,
            epoch_num=1000 ,
            cs = False , 
            component_num = 10 ,
            num_train_batch = 2048,
            num_test_batch = 512 ,
            sum_to_zero=True,
            init_train=True,
            reg_lambda=0.0,
            features_list="all",
            uniform_transform = True):
    in_features = data_x.shape[1]
    
    if regression:
        multiclass = 2

    
    if multiclass == 2:
        class_num = 1
    #else:
    #    num_multiclass = multiclass
        
        
    if regression == True:
        global_opt_loss = np.inf 
    else:
        global_opt_loss = -np.inf
        
    choice_function=nn_utils.entmax15
    bin_function=nn_utils.entmoid15

    input_dim = in_features
    num_features = input_dim

    layer_dim = tree_num    
    
    ### generating model
    features_list, layers = arch.gen_odst(num_features,input_dim,layer_dim,device,class_num,max_order,choice_function, bin_function,features_list=features_list)
    model = nn.Sequential(
        arch.DenseBlock(features_list,layers,device),
        nn_utils.Lambda(lambda x:  x.mean(dim=1)),
    )
    model = model.to(device)    
    
    if uniform_transform == True:
        data_x = 2*qt.fit_transform(data_x) -1
        
    train_x,test_x_,train__y,test_y_ = train_test_split(data_x,data_y, test_size=0.3, random_state=random_state)
    val_x,test_x,val_y,test_y = train_test_split(test_x_,test_y_, test_size=0.66, random_state=0)
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    val_x = scaler.transform(val_x)
    
        
    if regression == True:
        train_y = (train__y - torch.mean(train__y))/torch.std(train__y)
        test_y = (test_y - torch.mean(train__y))/torch.std(train__y)
        val_y = (val_y - torch.mean(train__y))/torch.std(train__y)
    else:
        
        if sum_to_zero == True:
            train_y = train__y - torch.mean(train__y)
        else:
            train_y = train__y 

    train_data = torch.cat([torch.tensor(train_x),train_y],dim=1)
    test_data = torch.cat([torch.tensor(test_x),test_y],dim=1)
    val_data = torch.cat([torch.tensor(val_x),val_y],dim=1)

    train_dataloader = DataLoader(train_data, batch_size=num_train_batch, shuffle=True,num_workers=10)
    test_dataloader = DataLoader(test_data, batch_size=num_test_batch, shuffle=True,num_workers=10)
    val_dataloader = DataLoader(val_data, batch_size=num_test_batch, shuffle=True,num_workers=10)  


    ############## Initalize ###############
    for train__ in train_dataloader:
        init_x,init_y = train__[:,:in_features].float() , train__[:,in_features].float()
        
        init_x,init_y = init_x.to(device),init_y.to(device)
        break
        
    #for batch_val in val_dataloader:
    #    batch_val = (batch_val[:,0:in_features], batch_val[:,in_features:])
    #    batch_val_x , batch_val_y = batch_val
    #    val__x = batch_val_x.float().to(device)
    #    val__y = batch_val_y.float().to(device)
        
    #for test__ in test_dataloader:
    #    test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)
        
    if init_train == True:    
        with torch.no_grad():
            model[0].training = True
            model[0](init_x,inital=True)
        
        
    optimizer = optim.Adam(list(model.parameters()), lr=lr_rate,betas=(0.95, 0.998))
    criterion = torch.nn.MSELoss(reduction='sum')
    
    if max_order == 1:
        interval_epoch = 30  
    else:
        interval_epoch = 100
    
       
    for epoch in range(epoch_num):
        
        model.train()
        loss_sum = 0
        model[0].training = True
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_ = (batch[:,0:in_features], batch[:,in_features:])
            batch_x , batch_y = batch_
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            
            if multiclass == 2:
                loss = criterion(output.flatten(),batch_y.flatten()).sum()
                
            else:
                print("multiclass > 3 error")
            
            #elif multiclass >= 3:
            #    loss = 0
            #    for c in range(0,multiclass):
            #        loss += criterion(output[:,c].flatten(),batch_y[:,c].flatten()).sum()
            if reg_lambda > 0.0 :
                #reg_loss = 0.0
                ##for i in range(0,len(features_list)):
                #    reg_loss += ( model[0](batch_x)[:,tree_num*i:tree_num*(i+1),-1].sum(axis=1) )**2 
                    
                #loss += reg_lambda*torch.mean(reg_loss)
                
                loss += reg_lambda*torch.mean( (model[0](batch_x))**2)
            
            loss.backward() 
            optimizer.step() 
            loss_sum += loss.item()
        
        epoch_RMSE = np.sqrt(loss_sum/len(train_data))*torch.std(train__y)


        model.eval()        
        model[0].training = False
        model[0].model_save_id_constants()
        
        
                   
        #Component Selection 하는 부분
        if cs == True:
            if epoch > 200:
                if epoch%interval_epoch == 0  :
                    

                    features_list,layers = utils.check_valuable_nn(features_list,layers,device,init_x,component_num )
                    model = nn.Sequential(
                        arch.DenseBlock(features_list,layers,device),
                        nn_utils.Lambda(lambda x:  x.mean(dim=1)),
                    )
                    
                    model = model.to(device)
                
        
        
        if regression == True:  
            val_loss =0
            for val__ in val_dataloader:
                val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
            
                val_loss += torch.sum( (model(val__x.float()).flatten() - val__y.flatten())**2 ).cpu().detach()
            val_rmse = np.sqrt(  (val_loss.cpu().detach() )/len(val_data)  )*torch.std(train__y)
            

                
            test_loss =0
            all_test__output = torch.tensor([])
            
            for test__ in test_dataloader:
                test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)   
            
                all_test__output = torch.concat([all_test__output, model(test__x.float()).cpu().detach()  ])
                
                test_loss += torch.sum( (model(test__x.float()).flatten() - test__y.flatten())**2 ).cpu().detach()
            test_local_measure = np.sqrt(  (test_loss.cpu().detach() )/len(test_data)  )*torch.std(train__y)
        
            
            #if test_local_measure < global_opt_loss :
            #    opt_epoch = epoch 
                
            #    global_opt_loss = test_local_measure
            
            if test_local_measure < global_opt_loss:
                opt_epoch = epoch
                global_opt_loss = test_local_measure
                
                test_measure = test_local_measure
                
                print(f"{epoch}번째 : train rmse : {epoch_RMSE } , val rmse : {val_rmse}, test rmse : {test_measure}")
    
                if model_path != None:
                    torch.save(model.state_dict(),model_path)
                    with open(f"{model_path}_features_list", "wb") as fp:   #Pickling
                        pickle.dump(features_list, fp)
                        
            if epoch - opt_epoch == 1000:
                return model,test_measure ,all_test__output
         
            
        else:
               
            if measure == "acc":
                val_hit =0
                for val__ in val_dataloader:
                    val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
                    
                    if sum_to_zero == True:
                        val_logit = model(val__x.float()).flatten() >= -torch.mean(train__y)
                        true_val_logit = val__y.flatten() >=0
                        
                        val_hit += torch.sum( val_logit== true_val_logit )
                    else:
                        val_logit = model(val__x.float()).flatten()  >= 0 
                        true_val_logit = val__y.flatten() >=0
                        
                        val_hit += torch.sum( val_logit== true_val_logit )                       
                
                val_measure = val_hit/ len( val_data )
                
            elif measure == "auc":
                    
                all_val__y = torch.tensor([])
                all_val__output = torch.tensor([])
                for val__ in val_dataloader:
                    val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
                    all_val__y = torch.concat([all_val__y,val__y.detach().cpu()])
                    all_val__output = torch.concat([all_val__output,(model(val__x.float()).reshape(-1,1)).detach().cpu()])
                      
                if sum_to_zero == True:
                    val_measure = roc_auc_score(all_val__y,all_val__output +  torch.mean(train__y))
                else:
                    val_measure = roc_auc_score(all_val__y,all_val__output )
                               
            
            if measure == "acc":
                test_hit = 0
                for test__ in test_dataloader:
                    test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)  
                    
                    if sum_to_zero == True:
                        test_logit = model(test__x.float()).flatten() >= -torch.mean(train__y)
                        true_test_logit = test__y.flatten() >=0
                        
                        test_hit += torch.sum( test_logit== true_test_logit )
                    else:
                        test_logit = model(test__x.float()).flatten()  >= 0 
                        true_test_logit = test__y.flatten() >=0
                        
                        test_hit += torch.sum( test_logit== true_test_logit )                                       

                test_candi_measure = test_hit/ len(test_data)
                
            elif measure == "auc":
                
                all_test__y = torch.tensor([])
                all_test__output = torch.tensor([])
                for test__ in test_dataloader:
                    test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)
                    all_test__y = torch.concat([all_test__y,test__y.detach().cpu()])
                    all_test__output = torch.concat([all_test__output,(model(test__x.float()).reshape(-1,1)).detach().cpu()]) 
               
                if sum_to_zero == True:
                    test_candi_measure = roc_auc_score(all_test__y,all_test__output + torch.mean(train__y))
                else:
                    test_candi_measure = roc_auc_score(all_test__y,all_test__output )            
                                       
            
            if test_candi_measure >= global_opt_loss :
                opt_epoch = epoch 
                global_opt_loss = test_candi_measure
                test_measure = global_opt_loss
                
                print(f"{epoch}번째 : train rmse : {epoch_RMSE } , val {measure} : {val_measure}, test {measure} : {test_measure}") 
                if model_path != None:
                    torch.save(model.state_dict(),model_path) 
                    with open(f"{model_path}_features_list", "wb") as fp:   #Pickling
                        pickle.dump(features_list, fp)
                        
            print(f"{epoch}번째 : train rmse : {epoch_RMSE }")
            if epoch - opt_epoch == 1000:
                return model,test_measure,all_test__output
                    
    return model,test_measure, all_test__output
    
    

def GAM_NODE_Trainer(data_x,data_y, tree_num , max_order , device, model_path , measure ,regression,random_state,multiclass = 2 ,lr_rate =0.01 , epoch_num=1000 , cs = False , component_num = 10 , num_train_batch = 2048, num_test_batch = 512 , sum_to_zero=True,init_train=True,reg_lambda=0.0):
    in_features = data_x.shape[1]
    
    if regression:
        multiclass = 2

    
    if multiclass == 2:
        class_num = 1
    #else:
    #    num_multiclass = multiclass
        
        
    if regression == True:
        global_opt_loss = np.inf 
    else:
        global_opt_loss = -np.inf
        
    choice_function=nn_utils.entmax15
    bin_function=nn_utils.entmoid15

    input_dim = in_features
    num_features = input_dim

    layer_dim = tree_num    
    
    ### generating model
    features_list, layers = arch.GAM_NODE_gen_odst(num_features,input_dim,layer_dim,class_num,max_order,choice_function, bin_function)
    model = nn.Sequential(
        arch.GAM_NODE_DenseBlock(features_list,layers,device),
        nn_utils.Lambda(lambda x:  x.mean(dim=1)),
    )
    model = model.to(device)    
    
    
    train_x,test_x_,train__y,test_y_ = train_test_split(data_x,data_y, test_size=0.3, random_state=random_state)
    val_x,test_x,val_y,test_y = train_test_split(test_x_,test_y_, test_size=0.66, random_state=0)
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    val_x = scaler.transform(val_x)

    if regression == True:
        train_y = (train__y - torch.mean(train__y))/torch.std(train__y)
        test_y = (test_y - torch.mean(train__y))/torch.std(train__y)
        val_y = (val_y - torch.mean(train__y))/torch.std(train__y)
    else:
        
        if sum_to_zero == True:
            train_y = train__y - torch.mean(train__y)
        else:
            train_y = train__y 

    train_data = torch.cat([torch.tensor(train_x),train_y],dim=1)
    test_data = torch.cat([torch.tensor(test_x),test_y],dim=1)
    val_data = torch.cat([torch.tensor(val_x),val_y],dim=1)

    train_dataloader = DataLoader(train_data, batch_size=num_train_batch, shuffle=True,num_workers=10)
    test_dataloader = DataLoader(test_data, batch_size=num_test_batch, shuffle=True,num_workers=10)
    val_dataloader = DataLoader(val_data, batch_size=num_test_batch, shuffle=True,num_workers=10) 


    ############## Initalize ###############
    for train__ in train_dataloader:
        init_x,init_y = train__[:,:in_features].float() , train__[:,in_features].float()
        
        init_x,init_y = init_x.to(device),init_y.to(device)
        break
        
    #for batch_val in val_dataloader:
    #    batch_val = (batch_val[:,0:in_features], batch_val[:,in_features:])
    #    batch_val_x , batch_val_y = batch_val
    #    val__x = batch_val_x.float().to(device)
    #    val__y = batch_val_y.float().to(device)
        
    #for test__ in test_dataloader:
    #    test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)
        
    if init_train == True:    
        with torch.no_grad():
            #model[0].training = True
            model[0](init_x,inital=True)
        
    optimizer = optim.Adam(list(model.parameters()), lr=lr_rate,betas=(0.95, 0.998))
    criterion = torch.nn.MSELoss(reduction='sum')
    
    if max_order == 1:
        interval_epoch = 30  
    else:
        interval_epoch = 100
    
       
    for epoch in range(epoch_num):
        
        model.train()
        loss_sum = 0
        model[0].training = True
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_ = (batch[:,0:in_features], batch[:,in_features:])
            batch_x , batch_y = batch_
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            
            if multiclass == 2:
                loss = criterion(output.flatten(),batch_y.flatten()).sum()
                
            else:
                print("multiclass > 3 error")
            
            #elif multiclass >= 3:
            #    loss = 0
            #    for c in range(0,multiclass):
            #        loss += criterion(output[:,c].flatten(),batch_y[:,c].flatten()).sum()
            if reg_lambda > 0.0 :
                #reg_loss = 0.0
                ##for i in range(0,len(features_list)):
                #    reg_loss += ( model[0](batch_x)[:,tree_num*i:tree_num*(i+1),-1].sum(axis=1) )**2 
                    
                #loss += reg_lambda*torch.mean(reg_loss)
                
                loss += reg_lambda*torch.mean( (model[0](batch_x))**2)
            
            loss.backward() 
            optimizer.step() 
            loss_sum += loss.item()
        
        epoch_RMSE = np.sqrt(loss_sum/len(train_data))*torch.std(train__y)


        model.eval()        
        #model[0].training = False
        #model[0].model_save_id_constants()
        
        
                   
        #Component Selection 하는 부분
        if cs == True:
            if epoch > 200:
                if epoch%interval_epoch == 0  :
                    

                    features_list,layers = utils.check_valuable_nn(features_list,layers,device,init_x,component_num )
                    model = nn.Sequential(
                        arch.DenseBlock(features_list,layers,device),
                        nn_utils.Lambda(lambda x:  x.mean(dim=1)),
                    )
                    
                    model = model.to(device)
                
        
        
        if regression == True:  
            val_loss =0
            for val__ in val_dataloader:
                val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
            
                val_loss += torch.sum( (model(val__x.float()).flatten() - val__y.flatten())**2 ).cpu().detach()
            val_rmse = np.sqrt(  (val_loss.cpu().detach() )/len(val_data)  )*torch.std(train__y)
            

                
            test_loss =0
            all_test__output = torch.tensor([])
            
            for test__ in test_dataloader:
                test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)   
            
                all_test__output = torch.concat([all_test__output, model(test__x.float()).cpu().detach()  ])
                
                test_loss += torch.sum( (model(test__x.float()).flatten() - test__y.flatten())**2 ).cpu().detach()
            test_local_measure = np.sqrt(  (test_loss.cpu().detach() )/len(test_data)  )*torch.std(train__y)
        
            
            #if test_local_measure < global_opt_loss :
            #    opt_epoch = epoch 
                
            #    global_opt_loss = test_local_measure
            
            if test_local_measure < global_opt_loss:
                opt_epoch = epoch
                global_opt_loss = test_local_measure
                
                test_measure = test_local_measure
                
                print(f"{epoch}번째 : train rmse : {epoch_RMSE } , val rmse : {val_rmse}, test rmse : {test_measure}")
    
                if model_path != None:
                    torch.save(model.state_dict(),model_path)
                    with open(f"{model_path}_features_list", "wb") as fp:   #Pickling
                        pickle.dump(features_list, fp)
                        
            if epoch - opt_epoch == 1000:
                return model,test_measure 
         
            
        else:
               
            if measure == "acc":
                val_hit =0
                for val__ in val_dataloader:
                    val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
                    
                    if sum_to_zero == True:
                        val_logit = model(val__x.float()).flatten() >= -torch.mean(train__y)
                        true_val_logit = val__y.flatten() >=0
                        
                        val_hit += torch.sum( val_logit== true_val_logit )
                    else:
                        val_logit = model(val__x.float()).flatten()  >= 0 
                        true_val_logit = val__y.flatten() >=0
                        
                        val_hit += torch.sum( val_logit== true_val_logit )                       
                
                val_measure = val_hit/ len( val_data )
                
            elif measure == "auc":
                    
                all_val__y = torch.tensor([])
                all_val__output = torch.tensor([])
                for val__ in val_dataloader:
                    val__x,val__y = val__[:,:in_features].to(device) , val__[:,in_features].to(device)
                    all_val__y = torch.concat([all_val__y,val__y.detach().cpu()])
                    all_val__output = torch.concat([all_val__output,(model(val__x.float()).reshape(-1,1)).detach().cpu()])
                      
                if sum_to_zero == True:
                    val_measure = roc_auc_score(all_val__y,all_val__output +  torch.mean(train__y))
                else:
                    val_measure = roc_auc_score(all_val__y,all_val__output )
                               
            
            if measure == "acc":
                test_hit = 0
                for test__ in test_dataloader:
                    test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)  
                    
                    if sum_to_zero == True:
                        test_logit = model(test__x.float()).flatten() >= -torch.mean(train__y)
                        true_test_logit = test__y.flatten() >=0
                        
                        test_hit += torch.sum( test_logit== true_test_logit )
                    else:
                        test_logit = model(test__x.float()).flatten()  >= 0 
                        true_test_logit = test__y.flatten() >=0
                        
                        test_hit += torch.sum( test_logit== true_test_logit )                                       

                test_candi_measure = test_hit/ len(test_data)
                
            elif measure == "auc":
                
                all_test__y = torch.tensor([])
                all_test__output = torch.tensor([])
                for test__ in test_dataloader:
                    test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)
                    all_test__y = torch.concat([all_test__y,test__y.detach().cpu()])
                    all_test__output = torch.concat([all_test__output,(model(test__x.float()).reshape(-1,1)).detach().cpu()]) 
               
                if sum_to_zero == True:
                    test_candi_measure = roc_auc_score(all_test__y,all_test__output + torch.mean(train__y))
                else:
                    test_candi_measure = roc_auc_score(all_test__y,all_test__output )            
                                       
            
            if test_candi_measure >= global_opt_loss :
                opt_epoch = epoch 
                global_opt_loss = test_candi_measure
                test_measure = global_opt_loss
                
                print(f"{epoch}번째 : train rmse : {epoch_RMSE } , val {measure} : {val_measure}, test {measure} : {test_measure}") 
                if model_path != None:
                    torch.save(model.state_dict(),model_path) 
                    with open(f"{model_path}_features_list", "wb") as fp:   #Pickling
                        pickle.dump(features_list, fp)
                        
            print(f"{epoch}번째 : train rmse : {epoch_RMSE }")
            if epoch - opt_epoch == 1000:
                return model,test_measure
                    
    return model,test_measure, all_test__output