
import numpy as np
from glob import glob

def data_for_loader(path_to_all_datasets,set='train',batch_size=64,max_pat=None,split='70_15_15',stages=(1,2,3,4,5)):
    #make sure path ends with /*
    if path_to_all_datasets[-2:]!='/*' and path_to_all_datasets[-1]!='/':
        path_to_all_datasets = path_to_all_datasets+'/*'
    elif path_to_all_datasets[-2:]!='/*' and path_to_all_datasets[-1]=='/':
        path_to_all_datasets = path_to_all_datasets+'*'
        
    #initialize how many datasets are available
    datasets = glob(path_to_all_datasets+'/views/'+split+'/fixed_split/'+set)
    for i in range(len(datasets)):
        datasets[i] = datasets[i].split('/')[-5]
    num_datasets = len(datasets)
    
    #fetch all data paths
    data_list_per_set = [glob(path_to_all_datasets[:-1]+x+'/views/'+split+'/fixed_split/'+set+'/*') for x in datasets]
    if max_pat != None:
        for i in range(len(data_list_per_set)):
            data_list_per_set[i] = data_list_per_set[i][:max_pat]
    
    #fetch number of patients per dataset
    data_per_set = [len(data_list_per_set[x]) for x in range(len(data_list_per_set))]
    
    #make one list
    data_list_per_set = [item for sublist in data_list_per_set for item in sublist]
    
    #calculate the chance of a dataset to be selected
    chance_of_selection = [(0.5/num_datasets)+(0.5*(x/np.sum(data_per_set))) for x in data_per_set]
    
    #create chance vector for the full patients list
    chance_list = (np.array(chance_of_selection)/data_per_set).repeat(data_per_set)
    
    #create stage vector to ensure all stages are presented in batch
    start_label = np.repeat(np.array(stages),np.ceil(batch_size/len(stages)))
    #pop values if necesary
    while len(start_label)>batch_size:
        rand_pop = np.random.randint(0,high=len(start_label))
        start_label = np.delete(start_label,rand_pop)
    
    dataset_params = {'dataset_path_list': data_list_per_set,'dataset_chance_list':chance_list, 'dataset_label_list':start_label,'datasets':datasets,'selection_chance':chance_of_selection}
    return dataset_params
