path = '/Users/laabidn1/RetroDiffuser/data/uspto-50k/raw/train.csv'
rxns = open(path, 'r').readlines()
prods = []
data = {}

print(f'rxns {len(rxns)}\n')

for i, rxn in enumerate(rxns):
    rcts = rxn.strip().split('>>')[0]
    prod = rxn.strip().split('>>')[1]
    
    if prod in data.keys():
        data[prod].append(rcts)
    else:
        data[prod] = [rcts]

multiple = {prod:all_rcts for prod, all_rcts in data.items() if len(all_rcts)>1}
print(f'multiple {multiple}\n')
print(f'len(prods) {len(prods)}\n')

