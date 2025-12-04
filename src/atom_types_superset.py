test = ['Os', 'Mg', 'Li', 'Mn', 'Co', 'Cs', 'Rh', 'Ni', 'Ag', 'Pb', 'Nd', 'Sb', 'W', 'Cu', 'Br', 'H', 
        'I', 'Mo', 'P', 'C', 'Hg', 'Pd', 'Zn', 'Ca', 'Si', 'K', 'Eu', 'N', 'O', 'Cr', 'B', 'Se', 'Sc', 
        'F', 'Tl', 'Ba', 'Yb', 'Sr', 'Fe', 'Re', 'Ru', 'Ce', 'Zr', 'Ga', 'Ti', 'Sn', 'As', 'Na', 'Ir', 
        'Ge', 'In', 'S', 'Pt', 'Al', 'Cl', 'Au']

train = ['Br', 'C', 'Pt', 'Ag', 'Ce', 'Ba', 'W', 'Te', 'Si', 'Fe', 'Ru', 'Sn', 'Li', 'Os', 'Rb', 
            'La', 'Pd', 'B', 'H', 'Sc', 'Cr', 'Ge', 'Ar', 'Cd', 'Ga', 'Zr', 'Cs', 'Rh', 'S', 'Hf', 
            'Hg', 'Be', 'F', 'As', 'Au', 'Na', 'Nd', 'Ir', 'P', 'Ti', 'Sm', 'In', 'V', 'Eu', 'Sr', 
            'Co', 'Zn', 'Ca', 'Pr', 'Cu', 'Mg', 'Bi', 'Cl', 'Ta', 'Mn', 'Y', 'Pb', 'Mo', 'K', 'Sb', 
            'I', 'Yb', 'Tl', 'Se', 'Dy', 'O', 'Re', 'N', 'He', 'Ni', 'Xe', 'Al']

val = ['Ce', 'Co', 'Yb', 'Nd', 'Li', 'Al', 'W', 'Mo', 'Na', 'Si', 'Br', 'H', 'Tl', 'Ru', 'S', 'K', 
       'F', 'Re', 'Hg', 'N', 'Ni', 'Pd', 'Sb', 'In', 'Mn', 'Mg', 'Os', 'Rh', 'Fe', 'O', 'Pb', 'I', 
       'As', 'Ti', 'B', 'Sn', 'Zr', 'C', 'Bi', 'Ge', 'Pt', 'Se', 'Sm', 'Ar', 'P', 'Cu', 'Ba', 'Ag', 
       'Cr', 'Sc', 'Au', 'Cs', 'Ta', 'Zn', 'Cl', 'Ca']


test_included = [a for a in test if a not in train]

val_included = [a for a in val if a not in train]

print(f'test_included {test_included}\n')
print(f'val_included {val_included}\n')
