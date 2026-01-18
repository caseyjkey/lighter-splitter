# Lighter Image Processing Pipeline

This pipeline processes photos of beaded lighters taken on a blue background, extracts individual lighters, stages them professionally, and prepares them for your Django e-commerce site.

## Overview

The pipeline takes a phone photo of multiple lighters on a blue poster board/sheet and processes them through several stages:

1. **Extract** - Detects and extracts individual lighters from the photo
2. **Stage** - Applies professional staging with website-matching background
3. **Pair** - ⚠️ **REQUIRED:** Manually pair front/back images (rename to `X-1.png` and `X-2.png`)
4. **Name** - AI-powered naming and pricing (prevents duplicates automatically, only processes `-1` files, auto-renames `-2` files)
5. **Import** - Ready for Django database import

**Key Point:** Manual pairing (Step 3) is **required** before running the naming script. You must rename your images to follow the `-1`/`-2` pattern so the script knows which images are pairs.

## Prerequisites

- Python 3.8+
- Google Gemini API key (for AI naming) - Get one at https://aistudio.google.com/
- Blue poster board or sheet for photographing lighters

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Google Gemini API key in `namer.py`:
   - Edit `namer.py` and update the `API_KEY` variable with your key

## Directory Structure

```
lighter-splitter/
├── images/              # Input: Place your phone photos here
├── extracted_lighters/   # Output: Individual lighter images
├── staged_lighters/     # Output: Professionally staged images
├── contours.py          # Step 1: Extract lighters from photo
├── stage.py             # Step 2: Stage images
├── namer.py             # Step 4: AI naming and pricing
├── pair_images.py       # Optional: Auto-pair by similarity
└── requirements.txt     # Python dependencies
```

## Processing Workflow

### Step 1: Take Photos

Take photos of your lighters on a **blue poster board or sheet**:
- Place lighters on the blue background
- Take photos from above (top-down view)
- You can take multiple photos (e.g., front and back of each set)
- Save photos to the `images/` directory

**Tips:**
- Use a solid blue background (poster board or sheet)
- Ensure good lighting
- Keep lighters separated enough for detection
- Take front and back photos of each set

### Step 2: Extract Individual Lighters

Run the extraction script to detect and extract each lighter from your photos:

```bash
python contours.py
```

**What it does:**
- Reads all images from `images/` directory
- Detects lighters using contour detection (works with blue background)
- Extracts each lighter as a separate image
- Straightens and crops each lighter
- Saves to `extracted_lighters/` directory

**Output:** Files named like `photo1_lighter_01.png`, `photo1_lighter_02.png`, etc.

### Step 3: Stage Images

Apply professional staging with your website's background color:

```bash
python stage.py
```

**What it does:**
- Reads images from `extracted_lighters/`
- Applies perspective correction
- Places lighter on professional canvas (1200x1500px)
- Adds contact shadow for depth
- Uses website-matching background color (#F4F1ED)
- Saves to `staged_lighters/` directory

**Output:** Professionally staged images ready for e-commerce

### Step 4: Manually Pair Front/Back Images ⚠️ REQUIRED

**IMPORTANT:** You MUST manually pair your images before running the naming script. The `namer.py` script only processes files ending in `-1.png` and automatically renames their `-2.png` counterparts.

**Why manual pairing?**
- You photograph both front and back of each lighter
- The extraction process creates separate files that need to be matched
- The naming script requires the `-1`/`-2` pattern to identify pairs

**How to pair images:**

1. Open the `staged_lighters/` directory in your file manager

2. Identify which images are pairs (front and back of the same lighter)

3. Rename each pair using one of these patterns:

   **Option A: Number-based (recommended for many lighters)**
   - First pair: `1-1.png` (front) and `1-2.png` (back)
   - Second pair: `2-1.png` (front) and `2-2.png` (back)
   - Third pair: `3-1.png` (front) and `3-2.png` (back)
   - And so on...

   **Option B: Descriptive names**
   - `yellow-stripes-1.png` (front) and `yellow-stripes-2.png` (back)
   - `blue-dots-1.png` (front) and `blue-dots-2.png` (back)

4. **Important rules:**
   - The `-1` file is the **front/primary** image (will be processed by AI)
   - The `-2` file is the **back/secondary** image (will be auto-renamed)
   - Both files in a pair must have the **same prefix** (e.g., `1-1.png` and `1-2.png`)
   - The `namer.py` script will only process `-1` files

**Example:**
If you have 3 lighters with front/back photos:
```
Before pairing:
- photo1_lighter_01.png (front of lighter 1)
- photo2_lighter_01.png (back of lighter 1)
- photo1_lighter_02.png (front of lighter 2)
- photo2_lighter_02.png (back of lighter 2)
- photo1_lighter_03.png (front of lighter 3)
- photo2_lighter_03.png (back of lighter 3)

After manual pairing:
- 1-1.png (front)
- 1-2.png (back)
- 2-1.png (front)
- 2-2.png (back)
- 3-1.png (front)
- 3-2.png (back)
```

**Optional: Auto-pairing helper** (for special cases only):
If you have images that were already named by AI but not properly paired, you can use:
```bash
python pair_images.py
```
This uses visual similarity to find pairs, but **manual pairing is required for the main workflow**.

### Step 5: Name and Price Images

**Prerequisites:** Make sure you've completed Step 4 (manual pairing) first!

Use AI to generate names, categories, and prices for your lighters:

```bash
python namer.py
```

**What it does:**
- **Only processes `-1` files** (the front/primary image in each pair)
- Sends each `-1` image to Google Gemini AI
- Generates:
  - **Name**: 2-word poetic name inspired by Indigenous themes (e.g., "Feather-Sun")
  - **Category**: One of [Infinite-Path, Earths-Hue, Ancestral-Motifs, Traditional-Rhythms]
  - **Price**: Around $55 (range: $40-$70, based on complexity and craftsmanship)
- Renames the `-1` file: `Feather-Sun_Infinite-Path_55-1.png`
- **Automatically finds and renames** the corresponding `-2` file: `Feather-Sun_Infinite-Path_55-2.png`

**Example transformation:**
```
Before (manually paired):
- 1-1.png
- 1-2.png

After (AI processed):
- Feather-Sun_Infinite-Path_55-1.png
- Feather-Sun_Infinite-Path_55-2.png
```

**Note:** 
- The script automatically matches `-2` files to their `-1` counterparts
- If a `-2` file is missing, you'll get a warning but processing continues
- Already processed files (with prices) are skipped automatically
- **Duplicate prevention:** The script automatically checks all existing names and tells the AI which names to avoid, ensuring each name+category combination is unique (same name allowed if different category)

**Output Format:**
- Primary image: `Name_Category_Price-1.png`
- Secondary image: `Name_Category_Price-2.png`

**Note:** 
- The script skips already processed files (those with prices)
- It will reprocess old format files (without prices) to add pricing
- Free tier has rate limits (20 requests/day) - script handles retries automatically

### Step 6: Import to Django

Use the provided Django management command to import images:

```bash
python manage.py import_lighters /path/to/staged_lighters
```

See `DJANGO_PARSING_GUIDE.md` for detailed instructions on:
- Parsing filenames
- Django model setup
- Import script usage

## File Naming Convention

Final images follow this format:

```
Name_Category_Price-Side.png
```

**Example:**
- `Feather-Sun_Infinite-Path_55-1.png` (primary/front image)
- `Feather-Sun_Infinite-Path_55-2.png` (secondary/back image)

**Fields:**
- **Name**: Poetic 2-word name (hyphenated)
- **Category**: One of the 4 categories
- **Price**: Numeric price (40-70)
- **Side**: `-1` for primary, `-2` for secondary

## Configuration

### contours.py
- `INPUT_DIR`: Source directory for photos (default: `"images"`)
- `OUTPUT_DIR`: Where extracted lighters are saved (default: `"extracted_lighters"`)
- `MIN_AREA`: Minimum contour area to detect as lighter (default: 2000)

### stage.py
- `INPUT_DIR`: Source directory (default: `"extracted_lighters"`)
- `OUTPUT_DIR`: Staged images output (default: `"staged_lighters"`)
- `BG_COLOR`: Background color matching your website (default: `#F4F1ED`)

### namer.py
- `INPUT_DIR`: Directory with paired images (default: `"staged_lighters"`)
- `API_KEY`: Your Google Gemini API key
- Processes only files ending in `-1.png`

## Troubleshooting

### No lighters detected in contours.py
- Ensure you're using a **solid blue background**
- Check lighting - avoid shadows
- Increase `MIN_AREA` if lighters are too small
- Ensure lighters are separated enough

### AI naming fails
- Check your Google Gemini API key
- Free tier has 20 requests/day limit - script will retry automatically
- Check internet connection

### Images not pairing correctly
- **Manual pairing is required** - ensure you've renamed files to `X-1.png` and `X-2.png` pattern
- Check that both images in each pair have the same prefix (e.g., `1-1.png` and `1-2.png`)
- Verify that `-1` files exist (these are what get processed)
- The `namer.py` script will automatically find and rename `-2` files, but they must match the `-1` prefix

## Tips for Best Results

1. **Photography:**
   - Use consistent blue background
   - Good, even lighting
   - Top-down angle
   - Keep lighters separated

2. **Pairing (REQUIRED before naming):**
   - **You must manually pair images** before running `namer.py`
   - Use consistent naming: numbers (`1-1.png`, `1-2.png`) or descriptive names (`yellow-1.png`, `yellow-2.png`)
   - The `-1` file is the front/primary image (this gets processed by AI)
   - The `-2` file is the back/secondary image (auto-renamed to match)
   - Double-check that pairs are correctly matched before running namer.py
   - Both files in a pair must share the same prefix

3. **Naming:**
   - Run namer.py during off-peak hours to avoid rate limits
   - The script will retry on rate limit errors
   - Processed files are skipped automatically

## Support

For Django import questions, see `DJANGO_PARSING_GUIDE.md`.

For issues with image processing, check:
- File formats (PNG recommended)
- Directory permissions
- Python version compatibility

## 📚 Related Projects

- **[spirit-beads-backend](https://github.com/caseyjkey/spirit-beads-backend)** - Django e-commerce backend (live at thebeadedcase.com)
- **[spirit-beads-ui](https://github.com/caseyjkey/spirit-beads-ui)** - React 18 production frontend
