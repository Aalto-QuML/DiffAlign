# Trying to get the 7ck baseline to work here
- Main issue: we have a checkpoint, fastest way to release the code is to be able to sample again from the checkpoint and replicate the results
    - issue is code + data processing have changed since the checkpoint was created
    - find the samples we used to get the numbers reported in the paper: [DONE]
        - [CONCLUSION] found samples from 7ck, epoch 720, (entire test set, 100 steps):
            - when recanonicalized in later rdkit version, we get slightly higher scores than what is reported in the wandb run/paper
            - the latest scores are computed with deduplicating again (normally we would have to merge the similar scores and rerank), instead we just take the first occurence of any duplicated samples.
        - can find runs by looking for 'eval_cond4' in wandb/retrodiffuser
        - note: smthg is off when trying to access the artifact from the wandb run, you can instead go to artifacts,
            look for the 7ck eval runs, and find the run producing the artifact of interest.
        - example runs:
            - https://wandb.ai/najwalb/retrodiffuser/runs/vlhf59si?nw=nwusernajwalaabid
            - https://wandb.ai/najwalb/retrodiffuser/runs/dtxq3uib?nw=nwusernajwalaabid
            - https://wandb.ai/najwalb/retrodiffuser/runs/8iz9iuq4?nw=nwusernajwalaabid
            - https://wandb.ai/najwalb/retrodiffuser/runs/v6kr7geh?nw=nwusernajwalaabid
            - https://wandb.ai/najwalb/retrodiffuser/runs/uarg8p8x?nw=nwusernajwalaabid
            - https://wandb.ai/najwalb/retrodiffuser/runs/xylrsza2?nw=nwusernajwalaabid
            - not sure why we didnt use these? maybe there was an issue in the implementation (i.e. some info leak?)?
            - samples=100, steps=100, epochs 480 to 520:
                - https://wandb.ai/najwalb/retrodiffuser/runs/imcqhl1m?nw=nwusernajwalaabid
            
            - samples=100, steps=100, epochs 420 to 460:
                - https://wandb.ai/najwalb/retrodiffuser/runs/gufpwkie?nw=nwusernajwalaabid
        - re evaluate the samples saved in these runs, see if can get the same topks with new and old code
    - see if can generate same samples with old code [PENDING]
        - [UPDATE] still not able to find a dataset config and code version that works.
        - normally the commit hash saved in the training or eval runs should work, might be a good place to start investigating.s
        - but not sure, ask severi
    - check all commits across all branches until we find one that works [PENDING]
        -  [UPDATE]: tried the following branches/commits (just ran the code with a new dataset folder, no extra investigation):
            - note: all branches share the same history (general git fact), and the latest branches all contain the commits that produced 7ck and its evals.
            - can e.g. look from adding_chiral_features branch. 
            - there is one commit with the message 'found THE BUG' but it also did not work off the bat
            - unclear what are the configs used in the actual runs because they sometimes get overriden in the run and not saved to wandb.
    - quick sample check (maybe from saved samples?)
    - run to check the final results across the test set
- Then we can refactor the code to match the working code use a debugger if needed
    - data processing is done
    - simplify diffusion code
    - clean neural network code
    - train, sample, and evaluate scripts

## Runs info:
- most likely wrong eval:
    - osfwk5os, 7mze65bk, jzqyl11v, g0psi3y7, 9xs78cq0
- most likely correct eval (complete 50k test set):
    - 



# Data processing
- Watch out for
    - supernode edges or not
    - aromatic/not bond types
    - train with non reconstructed atoms too
    - atom types? used with original checkpoint
- Ablations
    - charged/not charged atom types
    - aromatic/not bond types