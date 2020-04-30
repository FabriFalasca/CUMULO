import numpy as np
import os
import time

from PIL import Image

import src.cloudsat
import src.interpolation
import src.modis_level1
import src.modis_level2
import src.tile_extraction

def extract_full_swath(myd02_filename, myd03_dir, myd06_dir, myd35_dir, cloudsat_lidar_dir, cloudsat_dir, verbose=1):
    """
    :param myd02_filename: the filepath of the radiance (MYD02) input file
    :param myd03_dir: the root directory of geolocational (MYD03) files
    :param myd06_dir: the root directory of level 2 files
    :param myd35_dir: the root directory to cloud mask files
    :param cloudsat_lidar_dir: the root directory of cloudsat-lidar files
    :param cloudsat_dir: the root directory of cloudsat files
    :param verbose: verbosity switch: 0 - silent, 1 - verbose
    :rtype: numpy.ndarray, dict, str, str 
    """

    filename = os.path.basename(myd02_filename)

    # pull a numpy array from the hdfs
    np_swath = src.modis_level1.get_swath(myd02_filename, myd03_dir)

    if verbose:
        print("swath {} loaded".format(filename))

    # as some bands have artefacts, we need to interpolate the missing data - time intensive
    t1 = time.time()
    
    filled_ch_idx = src.interpolation.fill_all_channels(np_swath)  
    
    t2 = time.time()

    if verbose:
        print("Interpolation took {} s".format(t2-t1))
        print("Channels", filled_ch_idx, "are now full")

    # tag current swath as "daylight" if all channels were filled
    if len(filled_ch_idx) == 15:
        tag = "daylight"

    # tag current swath as "night" if all but visible channels (not available during night) were filled
    elif filled_ch_idx == list(range(2, 7)) + list(range(8, 15)):
        tag = "night"

    # tag current swath as "corrupt" if data is missing for other reasons
    else:
        tag = "corrupt"

    # pull L2 channels here
    l2_channels = src.modis_level2.get_channels(myd02_filename, myd06_dir)
    
    if verbose:
        print("Level2 channels loaded")

    # pull cloud mask channel
    cm = src.modis_level2.get_cloud_mask(myd02_filename, myd35_dir)

    if verbose:
        print("Cloud mask loaded")

    # get cloudsat alignment - time intensive
    t1 = time.time()

    try:
        # alignment returns:
        # cs_range: minimal and maximal column indices of the satellite track for the current swath 
        # mapping: cloudsat-pixels -> swath pixels
        # cloudsat_info: available cloudsat variable values for the current swath
        cs_range, mapping, cloudsat_info = src.cloudsat.get_cloudsat_mask(myd02_filename, cloudsat_lidar_dir, cloudsat_dir, np_swath[-2], np_swath[-1], map_labels=False)

    except Exception as e:

        print("Couldn't extract cloudsat track of {}: {}".format(filename, e))

    t2 = time.time()

    if verbose:
        print("Cloudsat alignment took {} s".format(t2 - t1))

    # cast swath values to float
    np_swath = np.vstack([np_swath, l2_channels, cm[None, ]]).astype(np.float16)
    
    try:
        cloudsat_info.update({"width-range": cs_range, "mapping": mapping})

    except:
        
        cloudsat_info = None

    return np_swath, cloudsat_info, tag, filename

def extract_swath_rbg(radiance_filepath, myd03_dir, save_dir, verbose=1):
    """
    :param radiance_filepath: the filepath of the radiance (MYD02) input file
    :param myd03_dir: the root directory of geolocational (MYD03) files
    :param save_dir:
    :param verbose: verbosity switch: 0 - silent, 1 - verbose
    :return: none
    Generate and save RBG channels of the given MYDIS file. Expects to find a corresponding MYD03 file in the same directory. Comments throughout
    """

    basename = os.path.basename(radiance_filepath)

    # creating the save subdirectory
    save_dir = os.path.join(save_dir, "rgb")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    visual_swath = src.modis_level1.get_swath_rgb(radiance_filepath, myd03_dir)
    
    #interpolate to remove NaN artefacts
    filled_ch_idx = src.interpolation.fill_all_channels(visual_swath)

    if len(filled_ch_idx) == 3:

        pil_loaded_visual_swath = Image.fromarray(visual_swath.transpose(1, 2, 0).astype(np.uint8), mode="RGB")

        save_filename = os.path.join(save_dir, basename.replace(".hdf", ".png"))
        pil_loaded_visual_swath.save(save_filename)

        if verbose > 0:
            print("RGB channels saved as {}".format(save_filename))

    else:
        print("Failed to interpolate RGB channels of", basename)

def extract_tiles_from_swath(np_swath, swath_name, save_dir, tile_size=3, verbose=1):
    # sample the swath for a selection of tiles and its associated metadata
    try: 
        label_tiles, nonlabel_tiles, label_metadata, nonlabel_metadata = src.tile_extraction.sample_labelled_and_unlabelled_tiles(np_swath, tile_size=tile_size)

    except ValueError as e:
        print("Tiles failed to extract.", str(e))
        exit(0)

    if verbose > 0:
        print("{} tiles extracted from swath {}".format(len(label_tiles) + len(nonlabel_tiles), swath_name))

    label_tiles_savepath_str = os.path.join(save_dir, "label", "tiles")
    label_metadata_savepath_str = os.path.join(save_dir, "label", "metadata")

    nonlabel_tiles_savepath_str = os.path.join(save_dir, "nonlabel", "tiles")
    nonlabel_metadata_savepath_str = os.path.join(save_dir, "nonlabel", "metadata")

    # create the save filepaths for the payload and metadata, and save the npys
    for dr in [label_tiles_savepath_str, label_metadata_savepath_str, nonlabel_tiles_savepath_str, nonlabel_metadata_savepath_str]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    filename_npy = swath_name.replace(".hdf", ".npy")

    np.save(os.path.join(label_tiles_savepath_str, filename_npy), label_tiles, allow_pickle=False)
    np.save(os.path.join(label_metadata_savepath_str, filename_npy), label_metadata, allow_pickle=False)
    np.save(os.path.join(nonlabel_tiles_savepath_str, filename_npy), nonlabel_tiles, allow_pickle=False)
    np.save(os.path.join(nonlabel_metadata_savepath_str, filename_npy), nonlabel_metadata, allow_pickle=False)

def save_as_npy(np_swath, filename, save_subdir, verbose=1):
    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    swath_savepath_str = os.path.join(save_subdir, filename.replace(".hdf", ".npy"))

    np.save(swath_savepath_str, np_swath, allow_pickle=False)

    if verbose:
        print("swath saved as {}".format(swath_savepath_str))

# Hook for bash
if __name__ == "__main__":

    import sys

    from pathlib import Path

    from netcdf.npy_to_nc import save_as_nc
    from src.utils import get_file_time_info

    # parse command arguments for save location and radiance filename
    save_dir = sys.argv[1]    
    myd02_filename = sys.argv[2]
    
    root_dir, filename = os.path.split(myd02_filename)

    month, day = root_dir.split("/")[-2:]

    # get time info
    year, abs_day, hour, minute = get_file_time_info(myd02_filename)
    save_name = "A{}.{}.{}{}.nc".format(year, abs_day, hour, minute)

    # recursvely check if file exist in save_dir
    for _ in Path(save_dir).rglob(save_name):
        raise FileExistsError("{} already exist. Not extracting it again.".format(save_name))

    myd03_dir = os.path.join(root_dir, "MODIS", "data", "MYD03", "collection61", year, month, day)
    myd06_dir = os.path.join(root_dir, "MODIS", "data", "MYD06_L2", "collection61", year, month, day)
    myd35_dir = os.path.join(root_dir, "MODIS", "data", "MYD35_L2", "collection61", year, month, day)
    cloudsat_lidar_dir = os.path.join(root_dir, "CloudSatLidar")
    cloudsat_dir = os.path.join(root_dir, "CloudSat")

    # extract training channels, validation channels, cloud mask, class occurences if provided
    np_swath, cloudsat_info, tag, swath_name = extract_full_swath(myd02_filename, myd03_dir, myd06_dir, myd35_dir, cloudsat_lidar_dir, cloudsat_dir, save_dir=save_dir, verbose=0, save=False)

    # creating the save directories
    save_subdir = os.path.join(save_dir, tag)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)

    # save swath as netcdf
    save_as_nc(np_swath, cloudsat_info, swath_name, os.path.join(save_subdir, save_name))

    # save visible channels as png for visualization purposes
    extract_swath_rbg(myd02_filename, os.path.join(year, month, day), save_subdir, verbose=0)

