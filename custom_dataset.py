import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms
import random
from sklearn.model_selection import StratifiedShuffleSplit

class CustomDataset(Dataset):
    
    def __init__(self, img_path='/data/neuromark2/Data/ABCD/DTI_Data_BIDS/Raw_Data/',
                 label_file='final_qc_no_y_cleaned.csv', transform=None,
                 target_transform=None, train=True, valid=False, random_state=52):
        path = '/data/users1/schitikesi1/Myresearch/Data_W/merge_data/'
        print("Initializing CustomDataset...")
        self.img_path = img_path
        self.dirs = os.listdir(img_path)
      
        self.vars = pd.read_csv(path + label_file, index_col='src_subject_id',
                                usecols=['src_subject_id', 'tfmri_nb_all_beh_ctotal_mrt',
                                         'tfmri_nb_all_beh_ctotal_stdrt', 'tfmri_nb_all_beh_c0b_rate',
                                         'tfmri_nb_all_beh_c2b_rate'])
        self.vars.columns = ['tfmri_nb_all_beh_ctotal_mrt', 'tfmri_nb_all_beh_ctotal_stdrt',
                             'tfmri_nb_all_beh_c0b_rate', 'tfmri_nb_all_beh_c2b_rate']
        self.vars['new_score'] = self.vars['tfmri_nb_all_beh_ctotal_mrt']

        print("Loaded labels...")
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
        train_idx, self.test_idx = next(sss.split(np.zeros_like(self.vars),
                                                   self.vars.new_score.values))
        if train or valid:
            self.vars = self.vars.iloc[train_idx]
        else:
            test_vars = self.vars.iloc[self.test_idx]
        
        self.vars = self.vars.sort_index()    
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        print("CustomDataset initialized.")

    
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        subject_dir = os.path.join(self.img_path, self.dirs[idx])
        # Check if the subject directory exists
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"Subject directory {subject_dir} not found.")
        
        # Specify the path to the target file
        target_file_path = os.path.join(subject_dir, 'Baseline', 'dti', 'dti_FA', 'tbdti32ch_FA.nii.gz')
        
        # Check if the target file exists
        if not os.path.exists(target_file_path):
            raise FileNotFoundError(f"Target file {target_file_path} not found.")
        
        # Load the data
        img = nib.load(target_file_path).get_fdata()

        label = self.vars.iloc[idx]

        # Preprocess image data
        img = torch.tensor(img)
        img = (img - img.mean()) / img.std()
        if torch.sum(torch.isnan(img)) > 0:
            print(f'Custom dataset, {idx}')
            exit(-1)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, int(label['tfmri_nb_all_beh_ctotal_mrt'])

# Initialize and print the dataset
# dataset = CustomDataset()
# print("Dataset length:", len(dataset))

# # Fetch a few items from the dataset
# for i in range(3):
#     sample = dataset[i]
#     print("Sample:", sample)
