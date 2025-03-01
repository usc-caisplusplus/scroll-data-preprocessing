import os
import hydra
import threading
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import zarr

from omegaconf import DictConfig


def find_num_segments(scroll: DictConfig, cfg: DictConfig) -> int:
    """
    Find the number of segments in a scroll.

    Args:
        scroll (DictConfig): Configuration object for a scroll.
        cfg (DictConfig): Configuration object.

    Returns:
        Number of segments in the scroll.
    """
    if scroll.num_segments > 0:
        return scroll.num_segments
    else:
        scroll_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.comp_dataset_path, f"Scroll{scroll.scroll_id}/segments")
        if not os.path.exists(scroll_dir):
            print(f'ERROR: path does not exist - {scroll_dir}, returning 0')
            return 0

        # Get list of all directories in this folder
        segment_dirs = [d for d in os.listdir(scroll_dir) if os.path.isdir(os.path.join(scroll_dir, d))]
        return len(segment_dirs)



def prepare_datasets(
    fragment_ids: List[str], cfg: DictConfig
) -> Dict[str, Tuple[
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
    List[Tuple[int, int, int, int]]
]]:
    """
    Prepare the training and validation datasets by processing all fragments.

    Args:
        fragment_ids (List[str]): List of fragment identifiers.
        cfg (DictConfig): Configuration object.

    Returns:
        Dict containing:
            - "fragments": (custom_dict)
            - "segments": (custom_dict)
        Where custom_dict contains these keys/values:
            - train_images (List[np.ndarray]): List of training image patches.
            - train_masks (List[np.ndarray]): List of training mask patches. (is all 0's for segments)
            - valid_images (List[np.ndarray]): List of validation image patches.
            - valid_masks (List[np.ndarray]): List of validation mask patches. (is all 0's for segments)
            - valid_xyxys (List[Tuple[int, int, int, int]]): Coordinates for validation patches.
    """
    fragment_results = [None] * len(fragment_ids)
    threads = []

    def thread_target(idx: int, fragment_id: str):
        is_val = (fragment_id == cfg.validation_fragment_id)
        scroll_id = -1
        fragment_results[idx] = process_fragment_or_segment(fragment_id, is_val, scroll_id, cfg)

    for idx, frag_id in enumerate(fragment_ids):
        thread = threading.Thread(target=thread_target, args=(idx, frag_id))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Do similar logic as above but for segments, if specified in config
    scrolls = cfg.scrolls
    total_segments = 0
    if scrolls:
        for scroll in scrolls:
            total_segments += find_num_segments(scroll, cfg)
    
    # If there are segments to read
    segment_results = [None] * total_segments
    if total_segments > 0:
        # One thread is used for each scroll
        scroll_threads = []

        def scroll_thread_target(start_idx, scroll_id, num_segments, val_seg_mod):
            # Find all segment_ids in scroll_id folder
            scroll_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.comp_dataset_path, f"Scroll{scroll_id}/segments")
            if not os.path.exists(scroll_dir):
                print(f'ERROR: path does not exist - {scroll_dir}, returning None')
                return None

            # Get list of all directories in this folder
            segment_dirs = [d for d in os.listdir(scroll_dir) if os.path.isdir(os.path.join(scroll_dir, d))]

            # Randomly select num_segments of them
            selected_segment_dirs = []
            if num_segments > 0:
                selected_segment_dirs = random.sample(segment_dirs, num_segments)
            # Select all segments if num_segments is -1
            elif num_segments == -1:
                selected_segment_dirs = segment_dirs

            # Call function to process each segment
            idx = start_idx
            for segment_id in selected_segment_dirs:
                # Set to be validation based on the mod provided
                is_val = False
                if val_seg_mod and (idx + 1) % val_seg_mod == 0:
                    is_val = True
                # print(f'segment {segment_id} in scroll {scroll_id}, so val = {val_id}')
                segment_results[idx] = process_fragment_or_segment(segment_id, is_val, scroll_id, cfg)
                idx += 1
        

        val_seg_mod = cfg.get('validation_segments_mod', None)
        start_idx = 0
        for scroll in scrolls:
            scroll_id = scroll.scroll_id
            num_segments = scroll.num_segments
            scroll_thread = threading.Thread(target=scroll_thread_target, args=(start_idx, scroll_id, num_segments, val_seg_mod))
            start_idx += num_segments

            scroll_threads.append(scroll_thread)
            scroll_thread.start()

        for scroll_thread in scroll_threads:
            scroll_thread.join()

    print("Aggregating results from all fragments.")
    
    def aggregate_results(results):
        """Helper function to aggregate dataset results."""
        keys = ['train_images', 'train_masks', 'valid_images', 'valid_masks', 'valid_xyxys']
        aggregated = {key: [] for key in keys}
        
        for res in results:
            if res is not None:
                for key, data in zip(keys, res):
                    aggregated[key].extend(data)
        
        return aggregated

    # Initialize final results structure
    final_results = {
        'fragments': aggregate_results(fragment_results),
        'segments': aggregate_results(segment_results) if total_segments > 0 else {key: [] for key in ['train_images', 'train_masks', 'valid_images', 'valid_masks', 'valid_xyxys']}
    }

    return final_results



def process_fragment_or_segment(
    id: str, is_val: bool, scroll_id: int, cfg: DictConfig
) -> Optional[
    Tuple[
        List[np.ndarray], List[np.ndarray],
        List[np.ndarray], List[np.ndarray],
        List[Tuple[int, int, int, int]]
    ]
]:
    """
    Process a single fragment to extract training and validation patches.

    Args:
        id (str): Identifier for the fragment/segment.
        is_val (bool): Whether the fragment/segment is for validation.
        scroll_id (int): Identifier for the scroll. Default is -1 (is a fragment, not a segment).
        cfg (DictConfig): Configuration object.

    Returns:
        A tuple of training images, training masks, validation images, validation masks,
        and validation patch coordinates; or None if processing fails.
    """
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = ([] for _ in range(5))

    if scroll_id < 0:
        data_path = os.path.join(hydra.utils.get_original_cwd(), cfg.comp_dataset_path, f"Fragments/Fragment{id}")
        print('Processing fragment', id)
    else:
        data_path = os.path.join(hydra.utils.get_original_cwd(), cfg.comp_dataset_path, f"Scroll{scroll_id}/segments/{id}")
        print(f"Processing segment {id} (scroll {scroll_id}), is_val = {'True' if is_val else 'False'}")

    if not os.path.exists(data_path):
        print(f"Fragment/segment directory does not exist: {data_path}")
        return None


    # Define file names and paths (cache)
    cache_dir = os.path.join(data_path, 'cache')
    cache_path = os.path.join(cache_dir, 'data.zarr')

    use_cache = cfg.get('use_cache', False)
    dataset_names = ['train_images', 'train_masks', 'valid_images', 'valid_masks', 'valid_xyxys']
    os.makedirs(cache_dir, exist_ok=True)
    if use_cache and False:

        # Open or create a Zarr store
        store = zarr.DirectoryStore(cache_dir)
        root = zarr.group(store)

        # Check if all datasets exist
        if all(name in root for name in dataset_names):
            print(f"Loading cached processed data for {f'fragment {id}' if scroll_id < 0 else f'segment {id} (scroll {scroll_id})'}.")
            return tuple(root[name][:] for name in dataset_names)


    try:
        image, mask, fragment_mask = read_fragment_images_and_masks(
            id, scroll_id, data_path, cfg.start_idx, cfg.end_idx, cfg
        )
    except Exception as e:
        print(f"Aborted reading {f'fragment {id}' if scroll_id < 0 else f'segment {id} (scroll {scroll_id})'}: {e}")
        return None

    # Generate coordinates for tiling
    x1_list = list(range(0, image.shape[1] - cfg.tile_size + 1, cfg.stride))
    y1_list = list(range(0, image.shape[0] - cfg.tile_size + 1, cfg.stride))
    windows_set = set()

    for y in y1_list:
        for x in x1_list:
            fragment_tile = fragment_mask[y:y + cfg.tile_size, x:x + cfg.tile_size]
            if not np.any(fragment_tile == 0):  # Only valid fragments
                contains_ink = False
                if scroll_id < 0:
                    mask_tile = mask[y:y + cfg.tile_size, x:x + cfg.tile_size]
                    contains_ink = (not np.all(mask_tile < 0.05))
                if (is_val) or (contains_ink) or (scroll_id > 0):
                    for yi in range(0, cfg.tile_size, cfg.size):
                        for xi in range(0, cfg.tile_size, cfg.size):
                            y1 = y + yi
                            x1 = x + xi
                            y2 = y1 + cfg.size
                            x2 = x1 + cfg.size
                            image_patch = image[y1:y2, x1:x2]
                            mask_patch = mask[y1:y2, x1:x2, None]
                            if id != cfg.validation_fragment_id:
                                train_images.append(image_patch)
                                train_masks.append(mask_patch)
                            else:
                                coords = (x1, y1, x2, y2)
                                if coords not in windows_set:
                                    valid_images.append(image_patch)
                                    valid_masks.append(mask_patch)
                                    valid_xyxys.append(coords)
                                    windows_set.add(coords)
    if scroll_id < 0:
        print("Finished processing fragment:", id)
    else:
        print(f'Finished processing segment {id} (scroll {scroll_id})')

    # Save to cache
    if use_cache:
        store = zarr.DirectoryStore(cache_path)
        root = zarr.open(store, mode='w')

        for name in dataset_names:
            root.create_dataset(name, data=locals()[name], overwrite=True)

        print(f"\tSaved processed data for {f'fragment {id}' if scroll_id < 0 else f'segment {id} (scroll {scroll_id})'} to cache.")

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def read_fragment_images_and_masks(
    id: str, scroll_id: int, data_path: str, start_idx: int, end_idx: int, cfg: DictConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read multi-layer images and corresponding masks for a given fragment/segment.

    Args:
        id (str): Identifier for the fragment/segment.
        data_path (str): Full data path.
        start_idx (int): Starting layer index.
        end_idx (int): Ending layer index.
        cfg (DictConfig): Configuration object.

    Returns:
        A tuple of stacked image layers, normalized ink mask, and padded fragment mask.
    """
    images = []
    for i in range(start_idx, end_idx):
        if scroll_id < 0:
            tif_path = os.path.join(data_path, f"surface_volume/{i:02}.tif")
            jpg_path = os.path.join(data_path, f"surface_volume/{i:02}.jpg")
        else:
            tif_path = os.path.join(data_path, f"layers/{i:02}.tif")
            jpg_path = os.path.join(data_path, f"layers/{i:02}.jpg")

            tif_path2 = os.path.join(data_path, f"layers/{i:03}.tif")
            jpg_path2 = os.path.join(data_path, f"layers/{i:03}.jpg")

        if os.path.exists(tif_path):
            image = cv2.imread(tif_path, 0)
        elif os.path.exists(jpg_path):
            image = cv2.imread(jpg_path, 0)
        elif os.path.exists(tif_path2):
            image = cv2.imread(tif_path2, 0)
        elif os.path.exists(jpg_path2):
            image = cv2.imread(jpg_path2, 0)
        else:
            print(f"Image not found for layer {i:02} in {f'fragment {id}' if scroll_id < 0 else f'segment {id} (scroll {scroll_id})'}")
            continue

        # Pad image to be divisible by tile_size
        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size) % cfg.tile_size
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size) % cfg.tile_size
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image, 0, 200)
        images.append(image)

    if len(images) == 0:
        raise ValueError(f"No images found for {f'fragment {id}' if scroll_id < 0 else f'segment {id} (scroll {scroll_id})'}")

    # Stack images along the channel dimension
    images = np.stack(images, axis=2)

    # Flip channels if specified
    if id in cfg.flip_ids:
        images = images[:, :, ::-1]

    # Read ink mask if this is a fragment
    mask = None
    if scroll_id < 0:
        inklabel_files = os.path.join(data_path, 'inklabels.png')
        if os.path.exists(inklabel_files):
            mask = cv2.imread(inklabel_files, 0)
        else:
            print(f"Ink labels not found for fragment {id}, creating empty mask.")
            mask = np.zeros(images.shape[:2], dtype=np.uint8)
    else:
        # Create empty mask (since segment)
        mask = np.zeros(images.shape[:2], dtype=np.uint8)

    # Read and pad mask
    if scroll_id < 0:
        fragment_mask_path = os.path.join(data_path, 'mask.png')
    else:
        fragment_mask_path = os.path.join(data_path, f'{id}_mask.png')

    if os.path.exists(fragment_mask_path):
        fragment_mask = cv2.imread(fragment_mask_path, 0)
        pad0 = (cfg.tile_size - fragment_mask.shape[0] % cfg.tile_size) % cfg.tile_size
        pad1 = (cfg.tile_size - fragment_mask.shape[1] % cfg.tile_size) % cfg.tile_size
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        print(f"Mask not found for {f'fragment {id}' if scroll_id < 0 else f'segment {id} (scroll {scroll_id})'}, creating full mask.")
        fragment_mask = np.ones(images.shape[:2], dtype=np.uint8) * 255

    # Normalize ink mask to [0, 1]
    mask = mask.astype(np.float32) / 255.0

    return images, mask, fragment_mask