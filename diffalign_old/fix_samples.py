# This was a temporary file used for fixing up some buggy samples.txt files

directory = "experiments/trained_models/rxn_absorbing_masknoedge_product_and_sn_dummy15_loss_ce_smiles_pos_enc_no_clfgnum_gpus_2_simplebatching_16_lr_0.0002_lamtrain_5_total_epochs_700-seed24tzzqtm1/eval_noduplicates_ncond_test_4949_lambdatest_5_elborep_1_test"
filename = "samples_epoch320.txt"
import os
import re
file = os.path.join(directory, filename)

f = open(file, "r")
lines = f.readlines()
f.close()

regex_cond = r"^\(cond.*?>>([^:]*):"
conditions = {}
for line in lines:
    match = re.search(regex_cond, line)
    if match:
        conditions[match.group(1)] = [line]

# Revised regular expression to match the criteria
# This regex will match any part after '>>' and before the second single quote ('), 
# but only if the line does not start with '(cond'
regex_revised = r"(?<=\>\>)([^']*)(?=', \[)"

print(conditions.keys())

for line in lines:
    match_new = re.search(regex_revised, line)
    if match_new:
        conditions[match_new.group(1)].append(line)

newfilename = "samples_epoch320.txt"
file = os.path.join(directory, newfilename)
with open(file, "w") as f:
    for key in conditions.keys():
        f.writelines(conditions[key])
# write the lines in a new file

dir_for_wandb = directory
entity = "najwalb"
project = "retrodiffuser"
run_id = "4tzzqtm1"
epoch = 320

import wandb
with wandb.init(id=run_id, project=project, entity=entity, resume="allow") as run:
    # Move back to the original directory so that wandb.save works properly
    # wandb.save(os.path.join(dir_for_wandb, "modified_config.yaml"))
    if os.path.exists(os.path.join(dir_for_wandb, f'samples_epoch{epoch}.txt')): # This only saves the non-resorted samples? Some of them?
        wandb.save(os.path.join(dir_for_wandb, f'samples_epoch{epoch}.txt'))