import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from withoutbg import WithoutBG
from PIL import Image

# ================= CONFIG =================
IMAGE_DIR = "images"                  # folder with input images
OUTPUT_DIR = "output_lighters"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
USE_WITHOUTBG_API = False             # Set to True to use API, False for local
WITHOUTBG_API_KEY = None              # Set your API key if using API

MAX_SIZE = None                       # resize long edge to this (set to None for full resolution)
USE_GPU = False                        # Set to True if you have CUDA GPU available
IOU_MERGE_THRESH = 0.12               # merge overlapping fragments
VERTICAL_MERGE_THRESH = 25            # merge masks vertically aligned (pixels)

# Expected lighter size AFTER resize (for filtering)
# Note: If using full resolution, adjust these values proportionally
MIN_W, MAX_W = 50, 260
MIN_H, MAX_H = 80, 260
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Load SAM --------
if USE_GPU and torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Using CPU (set USE_GPU=True if you have a CUDA GPU)")

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=17,
    pred_iou_thresh=0.88,  # Lowered to capture more of the lighter
    stability_score_thresh=0.90,  # Lowered slightly
    crop_n_layers=1,
)

# -------- Load WithoutBG model --------
if USE_WITHOUTBG_API and WITHOUTBG_API_KEY:
    withoutbg_model = WithoutBG.api(api_key=WITHOUTBG_API_KEY)
    print("Using WithoutBG API")
else:
    withoutbg_model = WithoutBG.opensource()
    print("Using WithoutBG open-source model")

# -------- Helper functions --------
def resize_image(img, max_size):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0

def merge_masks(mask_dicts):
    """
    Merge masks that overlap or are vertically aligned (for connecting lighter body to head).
    """
    merged = []
    used = set()

    for i, m1 in enumerate(mask_dicts):
        if i in used:
            continue

        combined = m1["segmentation"].copy()
        x1, y1, w1, h1 = [int(v) for v in m1["bbox"]]
        center_x1 = x1 + w1 // 2
        bottom_y1 = y1 + h1

        for j, m2 in enumerate(mask_dicts):
            if j == i or j in used:
                continue

            x2, y2, w2, h2 = [int(v) for v in m2["bbox"]]
            center_x2 = x2 + w2 // 2
            top_y2 = y2

            # Check IOU overlap
            overlap = iou((x1, y1, w1, h1), (x2, y2, w2, h2)) > IOU_MERGE_THRESH
            
            # Check vertical alignment (one mask above the other, similar x-center)
            vertical_align = (
                abs(center_x1 - center_x2) < max(w1, w2) * 0.6 and  # Similar x position
                (abs(bottom_y1 - top_y2) < VERTICAL_MERGE_THRESH or  # Close vertically
                 abs(y1 - (y2 + h2)) < VERTICAL_MERGE_THRESH)
            )

            if overlap or vertical_align:
                combined = np.maximum(combined, m2["segmentation"])
                used.add(j)
                # Update bbox for next iteration
                x1 = min(x1, x2)
                y1 = min(y1, y2)
                w1 = max(x1 + w1, x2 + w2) - x1
                h1 = max(y1 + h1, y2 + h2) - y1
                center_x1 = x1 + w1 // 2
                bottom_y1 = y1 + h1

        merged.append(combined)

    return merged

def expand_mask(mask, iterations=6):
    """
    Expand mask, but prefer upward expansion (for lighter head).
    """
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    
    # Standard dilation
    expanded = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    
    # Additional aggressive upward expansion (for lighter head)
    # Create a kernel that expands more upward
    up_kernel = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=np.uint8)
    # More aggressive upward expansion
    expanded = cv2.dilate(expanded, up_kernel, iterations=6)
    
    # Also expand upward from the top edge of the mask
    ys, xs = np.where(expanded > 0)
    if len(ys) > 0:
        top_y = ys.min()
        # Get the width at the top
        top_xs = xs[ys == top_y]
        if len(top_xs) > 0:
            left_x = top_xs.min()
            right_x = top_xs.max()
            width = right_x - left_x
            
            # Extend upward by ~20% of the width (typical lighter head height)
            extend_up = max(8, int(width * 0.20))
            h, w = expanded.shape
            for i in range(1, extend_up + 1):
                y_extend = max(0, top_y - i)
                if y_extend >= 0:
                    # Extend the top edge upward
                    expanded[y_extend, left_x:right_x+1] = 255
    
    return expanded

def fill_mask(mask):
    """
    Safely fills holes inside a binary mask.
    Never crashes, even for tiny masks.
    """
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape

    # Pad by 1 pixel so background is guaranteed at (0,0)
    padded = cv2.copyMakeBorder(
        mask, 1, 1, 1, 1,
        cv2.BORDER_CONSTANT, value=0
    )

    flood = padded.copy()
    h2, w2 = flood.shape

    # Mask for floodFill must be (h+2, w+2)
    flood_mask = np.zeros((h2 + 2, w2 + 2), np.uint8)

    # This seed is ALWAYS valid
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    # Invert flood-filled background
    flood_inv = cv2.bitwise_not(flood)

    # Combine original + holes
    filled = padded | flood_inv

    # Remove padding
    return (filled[1:-1, 1:-1] > 0).astype(np.uint8)

def remove_background_withoutbg(image_path):
    """
    Use WithoutBG to remove background from the full input image.
    Returns the result as a numpy array (RGB) with alpha channel.
    """
    try:
        result = withoutbg_model.remove_background(image_path)
        # Convert PIL Image to numpy array
        if hasattr(result, 'convert'):
            # It's a PIL Image
            result_rgba = np.array(result.convert('RGBA'))
            return result_rgba
        else:
            # Already a numpy array or different format
            return np.array(result)
    except Exception as e:
        print(f"    Warning: WithoutBG failed for {image_path}: {e}")
        return None




# -------- Process images --------
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for img_name in image_files:
    print(f"Processing {img_name}...")

    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # Step 1: Remove background from full image using WithoutBG
    print(f"  Removing background with WithoutBG...")
    bg_removed = remove_background_withoutbg(img_path)
    
    alpha_channel = None
    if bg_removed is None:
        print(f"  Warning: WithoutBG failed, falling back to original image")
        # Fallback to original image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Convert RGBA to RGB for processing (we'll use alpha channel for final extraction)
        if bg_removed.shape[2] == 4:
            # Extract alpha channel for later use
            alpha_channel = bg_removed[:, :, 3]
            # Convert RGBA to RGB for SAM processing
            image_rgb = bg_removed[:, :, :3]
            # Convert to BGR for OpenCV operations
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_rgb = bg_removed
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Store original for full-res extraction (before any resizing)
    original_image_rgb = image_rgb.copy()
    original_shape = image_rgb.shape[:2]
    original_alpha = alpha_channel.copy() if alpha_channel is not None else None
    
    # Resize for detection (if MAX_SIZE is set)
    if MAX_SIZE is not None:
        print(f"  Resizing from {original_shape} to max {MAX_SIZE} for detection...")
        image_rgb = resize_image(image_rgb, MAX_SIZE)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Calculate scale factor
        scale_factor = MAX_SIZE / max(original_shape) if max(original_shape) > MAX_SIZE else 1.0
        
        # Resize alpha channel if we have it
        if alpha_channel is not None:
            h, w = original_shape
            alpha_channel = cv2.resize(alpha_channel, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"  Processing at full resolution: {original_shape}")
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        scale_factor = 1.0

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    # Step 2: Generate masks from background-removed image
    print(f"  Detecting lighters in background-removed image...")
    masks = mask_generator.generate(image_rgb)
    print(f"  Found {len(masks)} raw masks")

    # Remove giant background masks
    masks = [m for m in masks if m["area"] < img_area * 0.4]

    # Merge fragments
    merged_masks = merge_masks(masks)

    # Expand masks (more iterations to capture the head)
    expanded_masks = [expand_mask(m, iterations=7) for m in merged_masks]

    # Fill holes
    expanded_masks = [fill_mask(expanded_mask) for expanded_mask in expanded_masks]

    final_masks = []

    for mask in expanded_masks:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue  # skip empty masks

        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        if MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H:
            final_masks.append(mask)

    print(f"  {len(final_masks)} lighters after filtering")

    # Step 3: Extract individual lighters using masks
    print(f"  Extracting {len(final_masks)} lighters...")
    base = os.path.splitext(img_name)[0]

    for idx, mask in enumerate(final_masks):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
            
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        
        # Scale coordinates back to original resolution if we resized
        if MAX_SIZE is not None and scale_factor < 1.0:
            x0 = int(x0 / scale_factor)
            x1 = int(x1 / scale_factor)
            y0 = int(y0 / scale_factor)
            y1 = int(y1 / scale_factor)
            # Use original image for extraction
            extract_img = original_image_rgb
            extract_alpha = original_alpha
            extract_h, extract_w = original_shape
        else:
            extract_img = image_rgb
            extract_alpha = alpha_channel
            extract_h, extract_w = img_h, img_w
        
        # Add padding to prevent cut-offs (more padding at top for head)
        pad_x = max(2, int((x1 - x0) * 0.03))
        pad_y_bottom = max(2, int((y1 - y0) * 0.02))
        pad_y_top = max(5, int((y1 - y0) * 0.08))  # More padding at top for head
        
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y_top)  # More padding at top
        x1 = min(extract_w, x1 + pad_x)
        y1 = min(extract_h, y1 + pad_y_bottom)

        # Crop from original resolution image
        cropped_img_rgb = extract_img[y0:y1, x0:x1]
        cropped_img = cv2.cvtColor(cropped_img_rgb, cv2.COLOR_RGB2BGR)
        
        # If we have alpha channel from WithoutBG, use it; otherwise create mask from detection
        if extract_alpha is not None:
            # Use alpha channel from WithoutBG
            cropped_alpha = extract_alpha[y0:y1, x0:x1]
            
            # Convert to RGBA
            rgba = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = cropped_alpha
        else:
            # Fallback: create a simple mask (full opacity for cropped region)
            rgba = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255  # Full opacity

        # Save the result
        out_path = os.path.join(
            OUTPUT_DIR, f"{base}_lighter_{idx:02d}.png"
        )
        cv2.imwrite(out_path, rgba)

    print(f"  Saved {len(final_masks)} lighters from {img_name}\n")
