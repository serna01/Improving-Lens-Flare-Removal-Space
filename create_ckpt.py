#save a checkpoint to load on the remove_flare python script
import tensorflow as tf

# Specify the path to your saved model directory
# Make sure this directory contains a 'variables' subdirectory with checkpoint files
saved_model_dir = "experiments/trained_model"

# Load the model
model = tf.saved_model.load(saved_model_dir)

# Create a new checkpoint manager to save the model
ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, "experiments/trained_model/", max_to_keep=5)

# Save the model as a checkpoint and verify the save path
save_path = ckpt_manager.save()
print("Model saved to:", save_path)
