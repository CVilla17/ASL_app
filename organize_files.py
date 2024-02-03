import os
import pandas as pd
import string
from sklearn.model_selection import train_test_split

def make_alpha_subdirs(root):

    for letter in string.ascii_uppercase:
        os.makedirs(f"{root}\\{letter}")


def move_files(curr_dir,new_dir,classes):
    
    classes = pd.read_csv(classes)
    
    class_names = list(classes.columns.values)[1:]
    labels = classes[class_names].idxmax(axis=1).to_frame("label")
    filenames = classes["filename"].to_frame()
    new_df = filenames.join(labels)
    new_df["label"] = new_df["label"].str.strip()
    for row in new_df.itertuples(index=False):
        subfolder = row[1]
        filename = row[0]
        os.replace(f"{curr_dir}\\{filename}",f"{new_dir}\\{subfolder}\\{filename}")

def join_datasets():
    """
    Makes train,val,test separation then adds to existing train, val, test folders
    """

    new_dir = "C:\\Users\\Carlos Villa\\ML_self_study\\ASL_alphabet_model\\SigNN Character Database"
    final_dir = "C:\\Users\\Carlos Villa\\ML_self_study\\ASL_alphabet_model\\data"
    filedf = pd.DataFrame(columns=["filepath","category"])
    run_ix =0
    for cat in os.listdir(new_dir):
        for ix,file in enumerate(os.listdir(f'{new_dir}\\{cat}')):
            run_ix+=1
            filename = f'{new_dir}\\{cat}\\{file}'
            tempdf = pd.DataFrame({'filepath': filename, 'category':cat},index=[run_ix])
            filedf = pd.concat([filedf,tempdf])
    X_train, X_test, _, _ = train_test_split(
        filedf, filedf['category'],stratify=filedf['category'], test_size=0.4)

    X_test, X_val, _, _ = train_test_split(
            X_test, X_test['category'], stratify=X_test['category'], test_size=0.5)
    
    X_train['type'] = 'train'
    X_val['type'] = 'valid'
    X_test['type'] = 'test'

    fulldf = pd.concat([X_train,X_test,X_val])

    

    for i,row in fulldf.iterrows():
        category = row["category"]
        section = row["type"]
        path = row["filepath"]
        new_file = row["filepath"].split("\\")[-1]
        
        os.replace(f'{path}',f'{final_dir}\\{section}\\{category}\\{new_file}')

    


            


    
    
if __name__ == "__main__":
    # make_alpha_subdirs("C:\\Users\\Carlos Villa\\ML_self_study\\ASL_alphabet_model\\new_valid")
    # move_files("C:\\Users\\Carlos Villa\\ML_self_study\\ASL_alphabet_model\\valid","C:\\Users\\Carlos Villa\\ML_self_study\\ASL_alphabet_model\\new_valid", "valid/_classes.csv")
    
    # join_datasets()
    pass