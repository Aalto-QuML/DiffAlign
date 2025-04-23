import wandb

# Login to wandb
wandb.login()

# Initialize API
api = wandb.Api()

# Get the run containing your video
run = api.run("najwalb/retrodiffuser/7ckmnkvc")

# Download specific video from the path you showed
# Example for the first video in your list
video_path = "media/videos/sample_chains/epoch740_rxn0_1895132_f3b725b902b5ed4c786f.mp4"
file = run.file(video_path)
file.download(replace=True)