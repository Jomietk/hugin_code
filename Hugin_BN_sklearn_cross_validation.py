


import numpy as np
import pandas as pd
import sys
from pyhugin91 import *




    #%%
    
def score_domain(name_model,data_name,name_class,nb_folds=10):
    
    try:
        
        
        
        data=DataSet.parse_data_set(data_name, separator=',', error_handler=None)             
        domain = Domain.parse_domain(name_model)
        L_nodes=domain.get_nodes()
        
        #Creat random prior distribution
        L_ini= [np.random.randint(1,100,i.get_table().get_size()) for i in L_nodes]    
        
        #Read the data for cross validation
        X=pd.read_csv(data_name)
        y_sat=X[name_class]
        
        node_class=domain.get_node_by_name(name_class)      
        Accuracy=0 
        
        
        #Stratified shuffle split cross validation    
        cv = StratifiedShuffleSplit(n_splits=nb_folds, test_size=0.2, random_state=42)          
        for train_index, test_index in cv.split(X, y_sat): 
           
            #Set the same prior distribution for each fold
            for i in range(len(L_nodes)):
                 L_nodes[i].get_table().set_data(L_ini[i])     
         
         
            #Add train set to the model    
            for index in train_index:
                domain.add_cases(data, index,1)                          
            domain.compile()
 
          
            #Initialize experience table 
            for i in L_nodes:
                size_table=i.get_experience_table().get_size()    
                i.get_experience_table().set_data([1/size_table]*size_table)
                
             
            domain.learn_tables()
            
    
            #Add test set to the model   
            domain.set_number_of_cases(0)                 
            for index in test_index:
                domain.add_cases(data, index,1)   
                
            
            
            # calculation of the predictionn       
            L_predict=np.zeros(len(test_index))
            L_true=np.zeros(len(test_index))
            for i in range(0,domain.get_number_of_cases()):               
                domain.enter_case(i)
                L_true[i]=node_class.get_case_state(i)                                      
                node_class.retract_findings()
              
                domain.propagate()                           
                L_predict[i]=np.argmax([node_class.get_belief(i) for i in range(node_class.get_number_of_states())])              
                domain.initialize() 
          
            print(L_predict, L_true)
   
            Accuracy=Accuracy+sum(L_predict==L_true)/len(L_predict)
   
        
               
            domain.uncompile()
            domain.set_number_of_cases(0)
            
        
            
            
            
        #Delet the model and data for memory 
        domain.delete()
        data.delete()
        

        Accuracy=Accuracy/nb_folds


    except HuginException:
        print("A Hugin Exception was raised!")
        raise

    print("Accuracy= ",Accuracy)
    
    return(Accuracy)
    
