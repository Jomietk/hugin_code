

import numpy as np
import pandas as pd
import sys
from pyhugin91 import *




    #%%
    


def model_creation(name_data,name_class):
       
    data_=pd.read_csv(name_data)
    domain =Domain()
    data=DataSet.parse_data_set(name_data, separator=',')
    
    for i in range(data.get_number_of_columns()):
        name=data.get_column_name(i)
        col=data_[name]
        
        
        # Test if the variables are categorical else numerical
        if isinstance(col[0], str):     
            node = Node(domain, CATEGORY.CHANCE, KIND.DISCRETE, SUBTYPE.LABEL)           
            node.set_name(name)
            node.set_label(name)
            St=list(set(data_[name]))                 
            node.set_number_of_states(len(St))         
            for p in range(len(St)):
                node.set_state_label(p,St[p])
               
        else:
            node = Node(domain, CATEGORY.CHANCE, KIND.DISCRETE, SUBTYPE.NUMBER)
            node.set_name(name)
            node.set_label(name)
            St=np.sort(list(set(data_[name])))  
            node.set_number_of_states(len(St))      
            for p in range(len(St)):             
                node.set_state_value(p,St[p])
            
                
    
    #For Naive Bay
    node_class=domain.get_node_by_name(name_class)
    for i in domain.get_nodes():
        if i.get_name()!=name_class:
            i.add_parent(node_class)
     
    #Hierarchical naive Bay:
    #domain.add_cases(data,0,data.get_number_of_rows())
    #domain.learn_hnb_structure(target=domain.get_node_by_name(name_class))
 
    domain.compile()
    domain.save_as_net("NB.net") 
    domain.delete()