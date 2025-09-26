import pandas as pd
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import glob
import re

def load_constellation_patterns(patterns_path, verbose=False):
    """Load constellation pattern names and images from patterns folder"""
    pattern_files = list(patterns_path.glob("*_pattern.png"))
    constellation_data = {}
    
    for pattern_file in pattern_files:
        # Extract constellation name by removing '_pattern.png' suffix
        name = pattern_file.stem.replace('_pattern', '')
        
        # Load the pattern image (keep color for better star/line separation)
        pattern_image = cv2.imread(str(pattern_file), cv2.IMREAD_COLOR)
        if pattern_image is not None:
            constellation_data[name] = {
                'image': pattern_image,
                'path': pattern_file
            }
    
    if verbose:
        print(f"Loaded constellation patterns: {list(constellation_data.keys())}")
    
    return constellation_data

def find_sky_image(constellation_folder):
    """Find the sky image in the constellation folder"""
    # Look for .tif files first, then other image formats
    image_extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for ext in image_extensions:
        image_files = list(constellation_folder.glob(ext))
        if image_files:
            return image_files[0]  # Return first found image
    
    return None

def template_match_patch(sky_image, patch_image, threshold=0.7):
    """
    Perform template matching between patch and sky image
    Returns: (x, y, confidence) if match found, else (None, None, 0)
    """
    try:
        # Convert to grayscale if needed
        if len(sky_image.shape) == 3:
            sky_gray = cv2.cvtColor(sky_image, cv2.COLOR_BGR2GRAY)
        else:
            sky_gray = sky_image.copy()
            
        if len(patch_image.shape) == 3:
            patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)
        else:
            patch_gray = patch_image.copy()
        
        # Perform template matching
        result = cv2.matchTemplate(sky_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            # Calculate center coordinates
            patch_h, patch_w = patch_gray.shape
            center_x = max_loc[0] + patch_w // 2
            center_y = max_loc[1] + patch_h // 2
            return (center_x, center_y, max_val)
        else:
            return (None, None, max_val)
            
    except Exception as e:
        return (None, None, 0)

def extract_star_positions_from_pattern(pattern_image, verbose=False):
    """
    Extract star positions from a constellation pattern image
    Pattern images have white circular dots (stars) connected by green lines
    Returns list of (x, y) coordinates of star centers
    """
    # Convert BGR to HSV for better color separation
    if len(pattern_image.shape) == 3:
        hsv = cv2.cvtColor(pattern_image, cv2.COLOR_BGR2HSV)
        # Also work with grayscale version
        gray = cv2.cvtColor(pattern_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = pattern_image.copy()
        hsv = cv2.cvtColor(cv2.cvtColor(pattern_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    
    # Method 1: Detect white circular regions (stars) using HoughCircles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,  # Minimum distance between circle centers
        param1=50,   # Edge detection threshold
        param2=15,   # Accumulator threshold (lower = more circles detected)
        minRadius=3, # Minimum circle radius
        maxRadius=15 # Maximum circle radius
    )
    
    star_positions = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            star_positions.append((x, y))
        
        if verbose:
            print(f"HoughCircles detected {len(star_positions)} circular stars")
    
    # Method 2: If HoughCircles fails, fall back to white pixel detection
    # but filter out green line pixels
    if len(star_positions) == 0:
        # Create mask for white pixels (high brightness, low saturation)
        if len(pattern_image.shape) == 3:
            # Convert to HSV for better white detection
            lower_white = np.array([0, 0, 200])    # Low saturation, high brightness
            upper_white = np.array([180, 30, 255])  # Allow some saturation variation
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
        else:
            # Grayscale fallback
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Remove green line pixels if we're working with color image
        if len(pattern_image.shape) == 3:
            # Create mask for green pixels (constellation lines)
            lower_green = np.array([40, 50, 50])    # Green hue range
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Remove green pixels from white mask
            white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(green_mask))
        
        # Find contours of white regions (stars)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area to avoid tiny noise
            area = cv2.contourArea(contour)
            if area > 10:  # Minimum area threshold
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    star_positions.append((cx, cy))
        
        if verbose:
            print(f"Contour method detected {len(star_positions)} white regions")
    
    if verbose:
        print(f"Total stars extracted from pattern: {star_positions}")
    
    return star_positions

def calculate_geometric_similarity(coords1, coords2, tolerance=0.3):
    """
    Calculate geometric similarity between two sets of star coordinates
    Uses normalized distances and angles between stars
    
    Args:
        coords1, coords2: Lists of (x, y) coordinates
        tolerance: Tolerance for matching (0.3 = 30% tolerance)
    
    Returns:
        similarity_score: Float between 0 and 1 (1 = perfect match)
    """
    if len(coords1) < 2 or len(coords2) < 2:
        return 0.0
    
    # Normalize coordinates to [0, 1] range
    def normalize_coords(coords):
        if not coords:
            return []
        xs, ys = zip(*coords)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Avoid division by zero
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        
        return [((x - min_x) / range_x, (y - min_y) / range_y) for x, y in coords]
    
    norm_coords1 = normalize_coords(coords1)
    norm_coords2 = normalize_coords(coords2)
    
    # Calculate pairwise distances for both sets
    def get_distances(coords):
        distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(dist)
        return sorted(distances)
    
    dist1 = get_distances(norm_coords1)
    dist2 = get_distances(norm_coords2)
    
    if not dist1 or not dist2:
        return 0.0
    
    # Compare distance patterns
    min_len = min(len(dist1), len(dist2))
    if min_len == 0:
        return 0.0
    
    # Calculate similarity based on distance ratios
    matches = 0
    for i in range(min_len):
        ratio = min(dist1[i], dist2[i]) / max(dist1[i], dist2[i]) if max(dist1[i], dist2[i]) > 0 else 0
        if ratio >= (1 - tolerance):
            matches += 1
    
    similarity = matches / min_len
    return similarity

def classify_constellation(valid_patches, constellation_data, sky_image_shape, verbose=False):
    """
    Classify constellation based on valid patch coordinates by comparing with reference patterns
    
    Args:
        valid_patches: List of (x, y) coordinates of valid patches
        constellation_data: Dict with constellation patterns and metadata
        sky_image_shape: Shape of the sky image for coordinate normalization
        verbose: Print debug information
    
    Returns:
        best_constellation: Name of best matching constellation or "unknown"
    """
    if not valid_patches or len(valid_patches) < 2:
        return "unknown"
    
    # Remove (-1, -1) coordinates
    valid_coords = [coord for coord in valid_patches if coord != (-1, -1)]
    
    if len(valid_coords) < 2:
        return "unknown"
    
    if verbose:
        print(f"Comparing {len(valid_coords)} valid coordinates against {len(constellation_data)} patterns")
    
    best_match = "unknown"
    best_score = 0.0
    scores = {}
    
    # Compare against each constellation pattern
    for constellation_name, pattern_data in constellation_data.items():
        pattern_image = pattern_data['image']
        
        # Extract star positions from pattern
        pattern_coords = extract_star_positions_from_pattern(pattern_image, verbose)
        
        if len(pattern_coords) < 2:
            continue
        
        # Calculate geometric similarity
        similarity = calculate_geometric_similarity(valid_coords, pattern_coords)
        scores[constellation_name] = similarity
        
        if similarity > best_score:
            best_score = similarity
            best_match = constellation_name
        
        if verbose:
            print(f"  {constellation_name}: {similarity:.3f} (pattern has {len(pattern_coords)} stars)")
    
    # Only return a match if the score is above a reasonable threshold
    min_threshold = 0.4  # Require at least 40% similarity
    if best_score < min_threshold:
        if verbose:
            print(f"Best match {best_match} with score {best_score:.3f} below threshold {min_threshold}")
        return "unknown"
    
    if verbose:
        print(f"Best match: {best_match} (score: {best_score:.3f})")
        print(f"All scores: {scores}")
    
    return best_match

def process_constellation_folder(constellation_folder, constellation_data, verbose=False):
    """Process a single constellation folder"""
    folder_name = constellation_folder.name
    
    if verbose:
        print(f"\nProcessing folder: {folder_name}")
    
    # Find sky image
    sky_image_path = find_sky_image(constellation_folder)
    if not sky_image_path:
        if verbose:
            print(f"No sky image found in {folder_name}")
        return {}, "unknown"
    
    # Load sky image
    sky_image = cv2.imread(str(sky_image_path))
    if sky_image is None:
        if verbose:
            print(f"Failed to load sky image: {sky_image_path}")
        return {}, "unknown"
    
    if verbose:
        print(f"Loaded sky image: {sky_image_path}")
        print(f"Sky image shape: {sky_image.shape}")
    
    # Find patches folder
    patches_folder = constellation_folder / "patches"
    if not patches_folder.exists():
        if verbose:
            print(f"No patches folder found in {folder_name}")
        return {}, "unknown"
    
    # Get all patch files with natural sorting
    patch_files = list(patches_folder.glob("patch_*.png"))
    patch_files = sorted(patch_files, key=lambda x: natural_sort_key(x.name))
    if verbose:
        print(f"Found {len(patch_files)} patch files")
        if patch_files:
            print(f"Patch order: {[p.name for p in patch_files[:5]]}{'...' if len(patch_files) > 5 else ''}")
    
    # Process each patch
    patch_results = {}
    valid_patches = []
    
    for patch_file in patch_files:
        patch_name = patch_file.name
        
        # Load patch image
        patch_image = cv2.imread(str(patch_file))
        if patch_image is None:
            patch_results[patch_name] = (-1, -1)
            continue
        
        # Perform template matching
        x, y, confidence = template_match_patch(sky_image, patch_image, threshold=0.6)
        
        if x is not None and y is not None:
            patch_results[patch_name] = (x, y)
            valid_patches.append((x, y))
            if verbose:
                print(f"  {patch_name}: ({x}, {y}) confidence={confidence:.3f}")
        else:
            patch_results[patch_name] = (-1, -1)
            if verbose:
                print(f"  {patch_name}: No match (confidence={confidence:.3f})")
    
    # Classify constellation using pattern matching
    constellation_prediction = classify_constellation(
        valid_patches, constellation_data, sky_image.shape, verbose
    )
    
    if verbose:
        print(f"Constellation prediction: {constellation_prediction}")
        print(f"Valid patches: {len(valid_patches)}/{len(patch_files)}")
    
    return patch_results, constellation_prediction

def natural_sort_key(text):
    """
    Generate a key for natural sorting (1, 2, 3, 4... not 1, 10, 11, 12...)
    """
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', str(text))]

def process_constellation_data(root_folder, target_folder, verbose=False):
    """
    Main processing function - implement your constellation classification here
    
    Args:
        root_folder: Path to root data directory (contains patterns/, test/, etc.)
        target_folder: Folder to process ('test', 'validation', etc.)
        verbose: Whether to print detailed output
    
    Returns:
        pandas.DataFrame: Results in required CSV format
    """
    
    root_path = Path(root_folder)
    target_path = root_path / target_folder
    patterns_path = root_path / "patterns"
    
    if verbose:
        print(f"Root folder: {root_path}")
        print(f"Target folder: {target_path}")
        print(f"Patterns folder: {patterns_path}")
    
    # Verify paths exist
    if not target_path.exists():
        raise FileNotFoundError(f"Target folder not found: {target_path}")
    if not patterns_path.exists():
        raise FileNotFoundError(f"Patterns folder not found: {patterns_path}")
    
    # Load constellation pattern data (images + names)
    constellation_data = load_constellation_patterns(patterns_path, verbose)
    
    # Find constellation folders with NATURAL SORTING
    constellation_folders = [f for f in target_path.iterdir() 
                           if f.is_dir() and not f.name.startswith('.')]
    # Use natural sorting to ensure constellation_1, constellation_2, constellation_3, etc.
    constellation_folders = sorted(constellation_folders, key=lambda x: natural_sort_key(x.name))
    
    if verbose:
        print(f"Found {len(constellation_folders)} constellation folders")
        if constellation_folders:
            print(f"Folder order: {[f.name for f in constellation_folders[:5]]}{'...' if len(constellation_folders) > 5 else ''}")
    
    # Process each constellation folder
    all_results = []
    max_patches = 0
    
    for i, constellation_folder in enumerate(constellation_folders, 1):
        folder_name = constellation_folder.name
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {i}/{len(constellation_folders)}: {folder_name}")
            print(f"{'='*50}")
        
        # Process constellation folder
        patch_results, constellation_prediction = process_constellation_folder(
            constellation_folder, constellation_data, verbose
        )
        
        # Track maximum patches across all folders
        max_patches = max(max_patches, len(patch_results))
        
        # Store results
        result_row = {
            "S.no": i,
            "Folder No.": folder_name,
            "patch_results": patch_results,
            "Constellation prediction": constellation_prediction
        }
        all_results.append(result_row)
    
    if verbose:
        print(f"\nMaximum patches found: {max_patches}")
    
    # Format results for CSV output
    for result in all_results:
        patch_results = result.pop("patch_results")
        
        # Sort patches by name using natural sorting (patch_01, patch_02, etc.)
        def extract_patch_number(patch_name):
            match = re.search(r'patch_(\d+)', patch_name)
            return int(match.group(1)) if match else 0
        
        # Use natural sorting for patches as well
        sorted_patches = sorted(patch_results.items(), key=lambda x: natural_sort_key(x[0]))
        
        # Add patch columns (patch 1, patch 2, ..., patch N)
        for patch_idx in range(1, max_patches + 1):
            col_name = f"patch {patch_idx}"
            
            if patch_idx <= len(sorted_patches):
                patch_name, (x, y) = sorted_patches[patch_idx - 1]
                if x == -1 and y == -1:
                    result[col_name] = "-1"
                else:
                    result[col_name] = f"({x},{y})"
            else:
                result[col_name] = "-1"
    
    # Create DataFrame with proper column order
    df = pd.DataFrame(all_results)
    
    # Ensure correct column order
    base_cols = ["S.no", "Folder No."]
    patch_cols = [f"patch {i}" for i in range(1, max_patches + 1)]
    final_cols = base_cols + patch_cols + ["Constellation prediction"]
    
    df = df.reindex(columns=final_cols)
    
    if verbose:
        print(f"\nFinal results shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Constellation Classification Assignment")
    parser.add_argument("root_folder", help="Root folder containing data and patterns")
    parser.add_argument("-f", "--folder", required=True,
                       help="Target folder to process (e.g., 'test', 'validation')")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Process the data
        results_df = process_constellation_data(
            args.root_folder,
            args.folder,
            args.verbose
        )
        
        # Save results in the same directory as this script
        script_dir = Path(__file__).parent
        output_file = script_dir / f"ap7151_{args.folder}_results.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"Results saved to: {output_file}")
        
        if args.verbose:
            print("\nSample output:")
            print(results_df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
