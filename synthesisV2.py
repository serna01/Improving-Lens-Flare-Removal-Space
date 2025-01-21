"""Processes paired flare and flare-free images for training."""
import tensorflow as tf
import utils

def process_paired_images(scene_with_flare, scene_without_flare, noise=0.0, training_res=512):
    """Processes paired images for training.

    Args:
        scene_with_flare: Image batch in sRGB with flare present.
        scene_without_flare: Corresponding image batch in sRGB without flare (ground truth).
        noise: Strength of the additive Gaussian noise.
        training_res: Resolution of training images.

    Returns:
        - Flare-free scene in sRGB (ground truth).
        - Flare-only image in sRGB.
        - Scene with flare in sRGB (input image).
        - Gamma value used during processing.
    """
    
    gamma = tf.random.uniform([], 1.8, 2.2)

    # Convert to linear space
    scene_with_flare_linear = tf.image.adjust_gamma(scene_with_flare, gamma)
    scene_without_flare_linear = tf.image.adjust_gamma(scene_without_flare, gamma)

    # Ensure the images are the correct size
    scene_with_flare_linear = tf.image.resize(scene_with_flare_linear, [training_res, training_res])
    scene_without_flare_linear = tf.image.resize(scene_without_flare_linear, [training_res, training_res])

    # Extract the flare by subtracting the flare-free scene from the scene with flare
    flare_linear = tf.clip_by_value(scene_with_flare_linear - scene_without_flare_linear, 0.0, 1.0)

    # Add noise to simulate sensor noise
    if noise > 0:
        sigma = tf.abs(tf.random.normal([], 0, noise))
        noise_tensor = tf.random.normal(scene_with_flare_linear.shape, 0, sigma)
        scene_with_flare_linear = tf.clip_by_value(scene_with_flare_linear + noise_tensor, 0.0, 1.0)

    # Convert back to sRGB space
    scene_without_flare_srgb = tf.image.adjust_gamma(scene_without_flare_linear, 1.0 / gamma)
    flare_srgb = tf.image.adjust_gamma(flare_linear, 1.0 / gamma)
    scene_with_flare_srgb = tf.image.adjust_gamma(scene_with_flare_linear, 1.0 / gamma)

    return (utils.quantize_8(scene_without_flare_srgb), 
            utils.quantize_8(flare_srgb),
            utils.quantize_8(scene_with_flare_srgb), 
            gamma)

def run_step(scene_with_flare,
             scene_without_flare,
             model,
             loss_fn,
             noise=0.0,
             training_res=512):
    """Executes a forward step."""
    # Process paired images
    scene_gt, flare, scene_with_flare, gamma = process_paired_images(
        scene_with_flare,  
        scene_without_flare,     
        noise=noise,
        training_res=training_res
    )

    # Model prediction (predicting the clean scene from the scene with flare)
    pred_scene = model(scene_with_flare)
    combined = scene_with_flare
    scene = scene_gt
    #pred_scene = model(combined)
    pred_flare = utils.remove_flare(combined, pred_scene, gamma)

    flare_mask = utils.get_highlight_mask(combined)
    # Ensure flare_mask has 3 channels by repeating the mask across the last dimension
    flare_mask = tf.tile(flare_mask, [1, 1, 1, 3])
    # Fill the saturation region with the ground truth, so that no L1/L2 loss
    # and better for perceptual loss since it matches the surrounding scenes.
    masked_scene = pred_scene * (1 - flare_mask) + scene * flare_mask
    loss_value = loss_fn(scene, masked_scene)

    # Calculate loss directly against the ground truth
    #loss_value = loss_fn(scene_gt, pred_scene)

    # Prepare image summary for TensorBoard
    image_summary = tf.concat([scene_with_flare,  # Original scene with flare
                               pred_scene,      # Model prediction
                               masked_scene,
                               scene_gt,        # Ground truth (scene without flare)
                               flare,
                               flare_mask],          # Actual flare
                              axis=2)

    return loss_value, image_summary
