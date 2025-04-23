import pickle

path = '/scratch/project_2006950/RetroDiffuser/data/uspto-mit-uncharged/processed/val.pickle'
train = pickle.load(open(path, 'rb'))
print(f'val {len(train)}\n')