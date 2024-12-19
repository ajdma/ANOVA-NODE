import torch
import torch.nn as nn
import torch.nn.functional as F

from .odst import ODST
from .odst import GAM_NODE_ODST

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy

from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from itertools import combinations

from models import nn_utils
import pickle

    
## Robust Interpretability
def cal_robust_interpretability_ver2(all_data_loader,h,cate_list, model_path, tree_num, max_order, in_features,device,regression,normalize = False):

    device = device
    choice_function=nn_utils.entmax15
    bin_function=nn_utils.entmoid15
        
    input_dim = in_features
    num_features = input_dim

    layer_dim = tree_num    
    multiclass = 2
    
    
    for data__ in all_data_loader:
        data__x = data__[:,:in_features].to(device) 
                        
    if regression:
        multiclass = 2

    if multiclass == 2:
        num_multiclass = 1
    else:
        num_multiclass = multiclass    
        
    var_features_list = []
    
    
    component_list = [i for i in range(0,in_features)]
    
    if max_order == 2:
        component_list.extend(list( combinations(component_list,2)))
    
    
    

    for l in range(0,len(component_list)):
        feature_j = component_list[l]
        
        print(f"Componet : {feature_j} processing...")
        
        all_diff_list2 = []
        for w in range(0,10):           


            features_list_cs, layers = gen_odst(num_features,input_dim,layer_dim,device,num_multiclass,max_order,choice_function, bin_function)
            
            model = nn.Sequential(
                DenseBlock(features_list_cs,layers,device),
                nn_utils.Lambda(lambda x:  x.mean(dim=1)),
            )
            model = model.to(device) 
            
            ##### 모형 불러오기 #####
            
            ### 저장된 parameter 불러오기

            load_model_state =  torch.load(model_path + f"-{w}") 
                        
            model.load_state_dict(load_model_state)     
            model[0].training = False   
            
            
            if type(feature_j) == int:
                all_pred_matrix = torch.tensor([])
                
                if w==0:
                    all_data_x = torch.tensor([])



                if w==0:
                    
                    if cate_list[feature_j] == "N" : 
                        h1=h
                        local_data__x1 = data__x[:,feature_j] - h1
                        local_data__x2 = data__x[:,feature_j] + h1
                    else:
                        h1=1
                        local_data__x1 = torch.tensor([1]).to(device)
                        local_data__x2 = torch.tensor([0]).to(device)                        
                        
                data_output_1 = model[0][l](local_data__x1.float().reshape(-1,1),False).mean(dim=1)
                data_output_2 = model[0][l](local_data__x2.float().reshape(-1,1),False).mean(dim=1)
                    
                all_pred_matrix = torch.concat([all_pred_matrix,data_output_1.reshape(-1,1).detach().cpu()])  
                all_pred_matrix = torch.concat([all_pred_matrix,data_output_2.reshape(-1,1).detach().cpu()],dim=1) 
                    
                denomi_x = h1
                nomi_f = all_pred_matrix[:,1] - all_pred_matrix[:,0]
                
                all_grad = nomi_f/denomi_x
                
                if normalize == True:
                    
                    if np.sum(np.array(all_grad)**2  ) !=0:
                        all_grad = all_grad / np.sqrt(np.sum(np.array(all_grad)**2  ))

                    
                all_diff_list2.append(all_grad.tolist() )
                
            else:
                if cate_list[feature_j[0]] == "N":
                    n_feature_1 = len(data__x)
                else:
                    n_feature_1 = 2

                if cate_list[feature_j[1]] == "N":
                    n_feature_2 = len(data__x)
                else:
                    n_feature_2 = 2
                    
                if n_feature_1 == 2 :
                    if n_feature_2 == 2:
                        all_data_x = torch.tensor([[1.0,1.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]).to(device)
                        all_pred_matrix = model[0][l](all_data_x.float(),False).mean(dim=1).detach().cpu().reshape(4,1)
                        
                        denomi_x  = 1
                        
                    else:
                        
                        for q in range(0,1):
                            data__x_new = data__x[:,feature_j]
                            data__x_new[:,0] = 0.5
                        
                            
                            all_data_x_1 = copy.deepcopy(data__x_new ) 
                            all_data_x_1[:,0] += 0.5
                            all_data_x_1[:,1] += h
                            
                            all_data_x_2 = copy.deepcopy(data__x_new ) 
                            all_data_x_2[:,0] += 0.5
                            all_data_x_2[:,1] -= h

                            all_data_x_3 = copy.deepcopy(data__x_new ) 
                            all_data_x_3[:,0] -= 0.5
                            all_data_x_3[:,1] += h

                            all_data_x_4 = copy.deepcopy( data__x_new ) 
                            all_data_x_4[:,0] -= 0.5
                            all_data_x_4[:,1] -= h            

                            all_pred_matrix = torch.concat([all_data_x_1,all_data_x_2,all_data_x_3,all_data_x_4])
                            all_pred_matrix = model[0][l](all_pred_matrix.float(),False).mean(dim=1).detach().cpu().reshape(4,len(data__x))
                            
                            denomi_x = 2*h
                            
                                
                else:
                    if n_feature_2 == 2:
                    
                        for q in range(0,1):
                            data__x_new = data__x[:,feature_j]
                            data__x_new[:,1] = 0.5
                             
                            
                            all_data_x_1 = copy.deepcopy( data__x_new ) 
                            all_data_x_1[:,0] += h
                            all_data_x_1[:,1] += 0.5
                            
                            all_data_x_2 = copy.deepcopy( data__x_new ) 
                            all_data_x_2[:,0] += h
                            all_data_x_2[:,1] -= 0.5

                            all_data_x_3 = copy.deepcopy( data__x_new ) 
                            all_data_x_3[:,0] -= h
                            all_data_x_3[:,1] += 0.5

                            all_data_x_4 = copy.deepcopy( data__x_new ) 
                            all_data_x_4[:,0] -= h
                            all_data_x_4[:,1] -= 0.5            

                            all_pred_matrix = torch.concat([all_data_x_1,all_data_x_2,all_data_x_3,all_data_x_4])
                            all_pred_matrix = model[0][l](all_pred_matrix.float(),False).mean(dim=1).detach().cpu().reshape(4,len(data__x))
                            
                            denomi_x = 2*h
                            
                    else:
                        data__x_new = torch.tensor([]).to(device)
                        for q in range(0,len(data__x)):
                            data_x_new_local = copy.deepcopy(data__x[:,feature_j])
                            
                            data_x_new_local[:,1] = data_x_new_local[q,1]
                            data__x_new = torch.concat([data__x_new,data_x_new_local])
                         
                        all_data_x_1 = copy.deepcopy( data__x_new ) 
                        all_data_x_1[:,0] += h
                        all_data_x_1[:,1] += h
                                
                        all_data_x_2 = copy.deepcopy( data__x_new ) 
                        all_data_x_2[:,0] += h
                        all_data_x_2[:,1] -= h

                        all_data_x_3 = copy.deepcopy( data__x_new ) 
                        all_data_x_3[:,0] -= h
                        all_data_x_3[:,1] += h

                        all_data_x_4 = copy.deepcopy( data__x_new ) 
                        all_data_x_4[:,0] -= h
                        all_data_x_4[:,1] -= h            

                        all_pred_matrix = torch.concat([all_data_x_1,all_data_x_2,all_data_x_3,all_data_x_4])
                        all_pred_matrix = model[0][l](all_pred_matrix.float(),False).mean(dim=1).detach().cpu().reshape(4,len(data__x)**2)    
                        
                        denomi_x =4*h*h 
                    
                nomi_f = all_pred_matrix[0,:] + all_pred_matrix[3,:] - all_pred_matrix[1,:] - all_pred_matrix[2,:]
                    
                all_grad = nomi_f/denomi_x
                    
                if normalize == True:
                    
                    if np.sum(np.array(all_grad)**2  ) != 0:
                        all_grad = all_grad / np.sqrt(np.sum(np.array(all_grad)**2  ))

                        
                all_diff_list2.append(all_grad.tolist() )
                    
        var_sum =0
        all_diff_list2  = np.array( all_diff_list2 )
        for c in range(0,all_diff_list2.shape[1]):
                
            var_sum += np.var(all_diff_list2[:,c])
                
        var_sum /= all_diff_list2.shape[1]
            
        var_features_list.append(var_sum)
                                        
    return var_features_list


## Robust Interpretability
def cal_UoC(all_data_loader, model_path, tree_num, max_order, in_features,device,regression,normalize = False,interaction_list = [],num_seed=10):

    device = device
    choice_function=nn_utils.entmax15
    bin_function=nn_utils.entmoid15
        
    input_dim = in_features
    num_features = input_dim

    layer_dim = tree_num    
    multiclass = 2
    
    
    for data__ in all_data_loader:
        data__x = data__[:,:in_features].to(device) 
                        
    if regression:
        multiclass = 2

    if multiclass == 2:
        num_multiclass = 1
    else:
        num_multiclass = multiclass    
        
    var_features_list = []
    abs_features_list = []
    
    if len(interaction_list) == 0:
        component_list = [i for i in range(0,in_features)]
        
        if max_order == 2:
            component_list.extend(list( combinations(component_list,2)))
    else:
        component_list = interaction_list
        
    
    model_list = []
    
    for w in range(0,num_seed):
        
        if len(interaction_list) == 0:
            features_list_cs, layers = gen_odst(num_features,input_dim,layer_dim,device,num_multiclass,max_order,choice_function, bin_function)
        else:
            features_list_cs, layers = gen_odst(num_features,input_dim,layer_dim,device,num_multiclass,max_order,choice_function, bin_function,features_list=interaction_list)
        model = nn.Sequential(
            DenseBlock(features_list_cs,layers,device),
            nn_utils.Lambda(lambda x:  x.mean(dim=1)),
        )
        model = model.to(device) 
        
        ##### 모형 불러오기 #####
        
        ### 저장된 parameter 불러오기

        load_model_state =  torch.load(model_path + f"-{w}") 
                    
        model.load_state_dict(load_model_state)     
        model[0].training = False 
        
        model_list.append(model)
    

    for l in range(0,len(component_list)):
        feature_j = component_list[l]
        
        print(f"Componet : {feature_j} processing...")
        
        all_output_trial = torch.tensor([])
        for w in range(0,num_seed):           
            
            
            if False:

                features_list_cs, layers = gen_odst(num_features,input_dim,layer_dim,device,num_multiclass,max_order,choice_function, bin_function)
                
                model = nn.Sequential(
                    DenseBlock(features_list_cs,layers,device),
                    nn_utils.Lambda(lambda x:  x.mean(dim=1)),
                )
                model = model.to(device) 
                
                ##### 모형 불러오기 #####
                
                ### 저장된 parameter 불러오기

                load_model_state =  torch.load(model_path + f"-{w}") 
                            
                model.load_state_dict(load_model_state)     
                model[0].training = False   
            else:
                model = model_list[w]
            
            if len( data__x[:,feature_j].shape ) ==1 :
                output_comp = model[0][l](data__x[:,feature_j].reshape(-1,1).float(),False).mean(dim=1).detach().cpu()
            else:
                
                if True:
                    input_new_data = torch.tensor([]).to(device)

                    for k in range(0,data__x.shape[0]):
                        
                        local_data = data__x[:,feature_j]
                        
                        local_data[:,0] = local_data[k,0]
                        
                        input_new_data = torch.concat([input_new_data,local_data])
                        
                    output_comp = model[0][l](input_new_data.float(),False).mean(dim=1).detach().cpu()
                else:
                    output_comp = model[0][l](data__x[:,feature_j].float(),False).mean(dim=1).detach().cpu()
                
            all_output_trial = torch.concat([all_output_trial,output_comp.reshape(1,-1)])
            
        if normalize == True:
                    
            for n in range(0,all_output_trial.shape[1]):
                if np.sqrt(np.sum(np.array(all_output_trial[:,n])**2))   != 0:
                    all_output_trial[:,n] = all_output_trial[:,n] /np.sqrt(np.sum(np.array(all_output_trial[:,n])**2))

                        
        var_sum =0
        all_output_trial  = np.array( all_output_trial )
        for c in range(0,all_output_trial.shape[1]):
                
            var_sum += np.var(all_output_trial[:,c])
                
        var_sum /= all_output_trial.shape[1]
        
        abs_sum = 0
        for c in range(0,all_output_trial.shape[1]):
                
            abs_sum += np.mean( np.abs( all_output_trial[:,c] - np.mean(all_output_trial[:,c]) ) )
                
        abs_sum /= all_output_trial.shape[1]
                
        var_features_list.append(var_sum)
        abs_features_list.append(abs_sum)
        
    return var_features_list,abs_features_list



############ Figure Main shape function ############  

def make_fig(data_x,data_y, regression,max_order,tree_num,columns_list,model_path,device,cs=False,fig=True,init_test=False,init_random_seed=0):
    train_x,test_x_,train__y,test_y_ = train_test_split(data_x,data_y, test_size=0.3, random_state=0)
    in_features = train_x.shape[1]
    
    for w in range(0,10):
    
        choice_function=nn_utils.entmax15
        bin_function=nn_utils.entmoid15
            
        input_dim = in_features
        num_features = input_dim

        layer_dim = tree_num    
        multiclass = 2
        
        
                            
        if regression:
            multiclass = 2

        if multiclass == 2:
            num_multiclass = 1
        else:
            num_multiclass = multiclass    
            
        if cs:
            feature_list_path = f"{model_path}-{w}_features_list"

            with open(feature_list_path, 'rb') as fp:
                features_list_cs = pickle.load(fp)
                
            
            print(f"{w}-th component set : {features_list_cs}")
            
            _,layers = gen_odst_cs(features_list_cs,layer_dim,num_multiclass,device,choice_function,bin_function)
            
        else:
            features_list_cs, layers = gen_odst(num_features,input_dim,layer_dim,device,num_multiclass,max_order,choice_function, bin_function)
        
        
        model = nn.Sequential(
                DenseBlock(features_list_cs,layers,device),
                nn_utils.Lambda(lambda x:  x.mean(dim=1)))
        

        ##### 모형 불러오기 #####
        
        ### 저장된 parameter 불러오기

        load_model_state =  torch.load(f"{model_path}-{w}")
        model.load_state_dict(load_model_state) 
        
        model = model.to(device)
        model.eval()
        
        if init_test == True:
            w = init_random_seed
            
        train_x,test_x_,train__y,test_y_ = train_test_split(data_x,data_y, test_size=0.3, random_state=w)
        val_x,test_x,val_y,test_y = train_test_split(test_x_,test_y_, test_size=0.66, random_state=0)
        
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        val_x = scaler.transform(val_x)
        
        if regression == True:
            test_y = (test_y - torch.mean(train__y))/torch.std(train__y)

        
        test_data = torch.cat([torch.tensor(test_x),test_y],dim=1)

        test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
        
        
        if regression == True:
            
            test_loss = 0
            for test__ in test_dataloader:
                test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)   
                
                #print(test__x,test__x.shape)

                test_loss +=  torch.sum( (model(test__x.float()).flatten() - test__y.flatten())**2 )
                
            test_rmse = torch.sqrt( test_loss.cpu().detach()/len(test_x) )
            print(f"state {w} 번째 : test rmse : {test_rmse*torch.std(train__y)}")
            
        else:
            all_test__y = torch.tensor([])
            all_test__output = torch.tensor([])
            for test__ in test_dataloader:
                test__x,test__y = test__[:,:in_features].to(device) , test__[:,in_features].to(device)
                all_test__y = torch.concat([all_test__y,test__y.detach().cpu()])
                all_test__output = torch.concat([all_test__output,(model(test__x.float()).reshape(-1,1)).detach().cpu()]) 
                
            test_measure = roc_auc_score(all_test__y,all_test__output)  
            print(f"state {w} 번째 : test auc : {test_measure}")
            
        ### 그림은 최대 10개까지만 그리는 걸로.. 설정함.
        ### main 개수 구하기
        
        if fig == True:
            main_list = []
            
            for f in range(0,len(features_list_cs)):
                
                if type( features_list_cs[f] ) == int:
                    
                    main_list.append(features_list_cs[f])
            
            
            if cs:
                max_feature = np.min([len(main_list),11])
            else:
                max_feature = np.min([in_features,11])
            
            
            f, axes = plt.subplots(1, max_feature, sharex=False, sharey=False)
            f.set_size_inches((25, 2))  
            
            f.text(0.09, 0.5, "Output Contribution", va='center', rotation='vertical') 
            
            y_max_list = []
            y_min_list = []
            for j in range(0,max_feature):
                y_max_list.append( np.max(np.array(  model[0][j](test__x[:,j].reshape(-1,1).float(),False).mean(dim=1).detach().cpu() ) ) )
                y_min_list.append( np.min(np.array(  model[0][j](test__x[:,j].reshape(-1,1).float(),False).mean(dim=1).detach().cpu() ) ) )
            
            
            y_max = np.max(y_max_list) + np.max(y_max_list)/10
            y_min = np.min(y_min_list) + np.min(y_min_list)/10
            
            sns.set(font_scale=1.0)
            for i in range(0,max_feature):
                
                axes[i].set(xlabel = columns_list[i])
            
                #axes[i].set(ylabel = "Output contribution")
                
                axes[i].set_ylim([y_min,y_max])
            
                scatter_x = np.array( test__x[:,i].detach().cpu() )

                scatter_y = np.array(  model[0][i](test__x[:,i].reshape(-1,1).float(),False).mean(dim=1).detach().cpu()  ) 
                
                #scatter = pd.DataFrame([scatter_x,scatter_y],index = [f"variable {i}",f"f {i}"]).T
                sns.lineplot(  x=scatter_x.flatten(),y=scatter_y.flatten(),ax=axes[i],color = "blue")
                sns.set(font_scale=1.0)
            f.show()