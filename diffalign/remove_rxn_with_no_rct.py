import os

directory = '/scratch/project_2006950/RetroDiffuser/data/uspto-full-uncharged/raw'

for f in ['train', 'test', 'val']:
    lines = open(os.path.join(directory, f'{f}.csv'), 'r').readlines()
    new_lines = [l for l in lines if l.split('>>')[0]!='']
    open(os.path.join(directory, f'{f}_new.csv'), 'w').writelines(new_lines)
