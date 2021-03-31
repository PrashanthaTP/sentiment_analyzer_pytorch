import os
import pandas as pd


CUR_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_location = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\sentanalyzer\sentanalyzer\data\src\aclImdb_v1.tar\aclImdb'
TEST_DIR = os.path.join(dataset_location,'train')
TRAIN_DIR = os.path.join(dataset_location,'test')

train_csv = os.path.join(CUR_DIR,r'movie_review_train.csv')
test_csv = os.path.join(CUR_DIR,r'movie_review_test.csv')

def create_csv(dir,filename):
    folders = os.listdir(dir)
    folders = [os.path.join(dir,folder) for folder in folders]
    folders = list(filter(os.path.isdir,folders))
    print(folders)
    df = pd.DataFrame()

    for folder in folders:
        label = os.path.basename(folder)
        limit = 0
        # path_to_folder = os.path.join(dir,folder)
        for f in os.listdir(folder):
            if limit%100==0:
                print(f'{os.path.join(folder,f)}')
            review = ''
            with open(os.path.join(folder,f),'r',encoding='utf-8') as fil:
                review = fil.read()
                review = review.strip()
                # review = review.lstrip('[')
                # review = review.rstrip(']')
                1
            df = df.append({'text':review,'sentiment':label},ignore_index=True) 
            limit+=1
            if limit>=1000:
                break 
    df.to_csv(filename)
    print(f"{filename} saved.")


def combine_train_test():
    train_df = pd.read_csv(r'sentanalyzer\test\movie_review_train.csv')
    test_df = pd.read_csv(r'sentanalyzer\test\movie_review_test.csv')
    train_df = train_df.append(test_df,ignore_index=True)
    train_df.to_csv(r'sentanalyzer/test/movie_review_train_test.csv')
    print('files combined and saved.')
def main():
    create_csv(TRAIN_DIR,train_csv)
    create_csv(TEST_DIR,test_csv)
    return
if __name__=='__main__':
    # main()
    combine_train_test()