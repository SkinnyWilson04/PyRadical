# Driver code to extract PyRadiomics features from the WT dataset, using the segmentation files emitted by FreeSurfer
import os
import SimpleITK as sitk                                # SimpleITK is a PyRadiomics dependency which handles image reads
import six                                              # six is a Python 2/3 compat library
import gzip                                             # Open compressed nifti files for use
from enum import Enum
import shutil
import yaml
from radiomics import featureextractor
from datetime import date
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog
from typing import List, Dict, TypeVar, Tuple


Arguments = TypeVar('argparse.Namespace')
Textfile  = TypeVar('_io.TextIOWrapper')


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Some handy-dandy error codes should we need to simply crash out of the program.
# This will be used to select what error information box is shown before the parent window is killed.

class ErrorCodes(Enum):
    SUCCESS = 0
    FAILURE = 1
    MISSINGVOLUME = 2
    MISSINGROI = 3
    MISSINGESSENTIALFILE = 4


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Error and success messages to display during program start-up. Messages are stored in lists to enable them to be printed
# via their index so that all errors can be reported, rather than crashing on the first error that is encountered (see
# perform_startup_checks() for usage)
STARTUP_FAILURE_MESSAGES: List[str] = ["Could not find path containing participant volumes / images",
                                       "Could not find path containing participant region-of-interest images",
                                       "Could not find path to maskvalues.txt file containing label:value information",
                                       "Could not find path to parameters.yaml file containing PyRadiomics params"]

STARTUP_SUCCESS_MESSAGES: List[str] = ["Good path to participant volumes / images!",
                                       "Good path to participant region-of-interest (ROI) files!",
                                       "Mask-values (index) text file found!",
                                       "PyRadiomics parameters YAML file found!"]


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Check that a directory or file at a given path actually exists. Returns True or False in each case rather than crashing
# to allow the caller to decide how to handle a missing file (i.e., may simply want to skip an interation, not exit).
# The error-message printed on failure can be suppressed with the 'suppress' flag (default is False).

def object_exists(path: str, suppress: bool = False) -> bool:
    try:
        assert os.path.exists(path) == True
        return True
    except AssertionError:
        if not suppress:
            print(f"Error: Cannot find file or directory '{path}'\n    -> Check filepath for this object and try again.")
        return False


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Returns a list of all of the object names in 'path' that are actually folders (or directories, if you prefer).
# (Please don't report me to IT Services for list comprehension abuse)

def list_only_foldernames(path: str) -> List[str]:
    folders: List[str] = [obj for obj in os.listdir(path) if os.path.isdir(os.path.join(path, obj))]
    return folders


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Simply reads the lines from a text file into a list of strings. Strips whitespace from each line-string before appending

def lines_from_file(filepath: str) -> List[str]:
    _ = object_exists(filepath)
    lines: List[str] = []
    
    with open(filepath) as fptr:
        for line in fptr:
            line = line.strip()
            if line:
                lines.append(line)

    return lines


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Reads lines from a file, where each line is a key:value pair separated by a specified separator. Used primarily to read
# index:name pairs for masks in a region-of-interest file (i.e., "1:amygdala", "14:hippocampus"), hence why the key:value
# types are assumed to be int and string, respectively. Whitespace is stripped from the mask name

def dict_from_file(filepath: str, separator) -> Dict[int, str]:
    _ = object_exists(filepath)
    values: Dict[int, str] = {}

    with open(filepath) as fptr:
        for line in fptr:
            key, value = line.split(separator)
            key = key.strip()
            values[int(key)] = value.strip()

    return values


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Get a value from a field in the extraction parameters yaml file
# First check that it exists, and then try to read the 'setting' dict from the data. If that works, try to read the
# 'field' from those settings. Assuming we didn't throw a KeyError, simply return the value for that field. If either of
# the keys are not found, then print the spurious key and return

def get_parameter_value(path: str, setting: str, field: str) -> any:
    if not object_exists(path):
        print("       Unable to find .yml file.")
        return

    with open(path, "r") as fptr:
        data = yaml.safe_load(fptr)

        try:
            field_value = data[setting][field]
            return field_value
        except KeyError as ke:
            print(f"! Key {ke} is not present in .yml file!")
            return            


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Attempt to find the requested volume (T1, ROI, etc.) file in the specified path. It's assumed that filenames will include
# the ID of participant from which they were collected, but that they may also have a consistent prefix, which is also
# handled here. If the requested file cannot be found, returns False; otherwise, returns True along with the full filepath

def resolve_target_filepath(path: str, prefix: str, ID: str) -> Tuple[bool, str]:
    print(f"{path}    {prefix}    {ID}")

    if prefix is None:
        prefix = ""
    
    files: List[str] = [file for file in os.listdir(path) if not os.path.isdir(os.path.join(path, file))]
    target: str = next(filter(lambda file: file.startswith(f"{prefix}{ID}"), files), None)
    fullpath: str = ""

    print(f"Target: {target}")
    
    if target is not None:
        fullpath: str = os.path.join(path, target)
        
        if os.path.exists(fullpath):
            print(f"    ! Found target file: '{target}'")
        else:
            print(f"    ! Error - Unable to find file at path '{fullpath}'")
            exit(3)
    else:
        print(f"    !!! Could not find requested file for ID '{ID}'")
        return (False, fullpath)
        
    return (True, fullpath)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >


def drop_unusable_IDs(IDs: List[str], folders: List[str], data_parent: str, check_for: str) -> (List[str], List[str]):
    remove_indices: List[int] = []
    for i, folder in enumerate(folders):
        data_location: str = os.path.join(parent, folder, "anat", folder, "mri", check_for)
        if not os.path.exists(data_location):
            remove_indices.append(i)
    folders.pop(remove_indices)
    IDs.pop(remove_indices)
    return(folders, IDs)
    

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >


def date_today() -> str:
    today_string: str = date.today().strftime("%d-%B-%Y")
    return today_string


def current_time() -> str:
    time_string: str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return time_string


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Appends the date of extraction and the bin-width specified in the 'parameters' .yaml file to the output folder
# name, to differentiate different runs and simplify runs with different parameters. Caller should overwrite the
# original 'args.output' string with this updated verbose output string

def verbose_output_location(where: str, parameters: str) -> str:
    date_str: str = date_today()
    binwidth: int = get_parameter_value(parameters, 'setting', 'binWidth')
    where_verbose = os.path.join(f"{where}", f"{date_str}_BinWidth_{str(binwidth)}")
    return where_verbose


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Creates an output folder in each of the 'foldernames' inside 'where', then creates two more subdirectories within that
# (for the left and right hemisphere-specific feature extraction outputs)
# First checks whether the output parent folder 'where' exists, and creates it if it does not (value for 'where'
# comes from the input argument 'args.output'

def create_output_folders(where: str, foldernames: List[str]):    
    print(f"Creating all output folders under: {where}")
    
    if os.path.exists(where):
        print(f"Note: The output folder '{where}' already exists!\n    -> Good to go!")
    else:
        os.mkdir(where)
        
    for folder in foldernames:
        out_folder: str = os.path.join(where, f"{folder}_Features")
        #l_folder:   str = os.path.join(out_folder, "LH")
        #r_folder:   str = os.path.join(out_folder, "RH")
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        #if not os.path.exists(l_folder):
        #    os.mkdir(l_folder)
        #if not os.path.exists(r_folder):
        #    os.mkdir(r_folder)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# gunzips 'filename_in' in the same folder ('where'). Needs to be done to all of the *.nii.gz files, as SimpleITK (which
# pyradiomics relies upon) simply won't read them otherwise. Returns the 'cleanup' list with the newly-decompressed file
# name appended to it to make sure that we're not missing anything that should have been deleted
def decompress_gz_here(where: str, filename_in: str, filename_out: str, cleanup: List[str]):
    filepath_in: str  = os.path.join(where, filename_in)
    filepath_out: str = os.path.join(where, filename_out)
    
    with gzip.open(filepath_in, 'rb') as file_in:
        with open(filepath_out, 'wb') as file_out:
            shutil.copyfileobj(file_in, file_out)
            
    cleanup.append(filepath_out)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Deletes the decompressed *.nii.gz files following feature extraction. 'cleanup_files' should contain the full path to
# each file to be deleted, not just the filename.
def cleanup_decompressed_gz(cleanup: List[str]):
    for file in cleanup:
        print(f"Note - Deleting decompressed '{file}', no longer needed.")
        os.remove(file)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Perform startup checks before any processing takes place. Report the current execution date, and then check that
# the paths to the mask-value file, volume and ROI files, and parameters file all exist and can be found.
# Report the result and then print a helpful error message if any of the checks fail, then exit the program
    
def perform_startup_checks():
    execution_date: str = date_today()
    path_volumes: str = input_volumes_entry.get()
    path_regions: str = input_regions_entry.get()
    path_maskvalues: str = input_maskvalues_entry.get()
    path_YAMLparams: str = input_YAMLparams_entry.get()

    paths_list: List[str] = [path_volumes, path_regions, path_maskvalues, path_YAMLparams]
    checkpath_results: List[bool] = [os.path.exists(user_path) for user_path in paths_list]

    print(f"[{execution_date}] Performing startup checks...")

    for i, checkresult in enumerate(checkpath_results):
        if checkresult:
            tk.messagebox.showinfo(title = "Startup Check",
                                   message = f"{paths_list[i]}\n{STARTUP_SUCCESS_MESSAGES[i]}")
        else:
            tk.messagebox.showerror(title = "Startup Failure",
                                    message = f"{STARTUP_FAILURE_MESSAGES[i]}\nConfirm path:\n{paths_list[i]}")


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Begin setup for feature extraction

def main():
    # Perform startup checks and ensure that the paths provided for the mask-value text file, image volumes
    # directory, ROI volumes directory, and the YAML parameters file are correct and can be found
    perform_startup_checks()
    
    # Read some required values into their own variables for ease of use
    output_location: str = output_location_entry.get()
    volumes: str = input_volumes_entry.get()
    regions: str = input_regions_entry.get()
    parameters: str = input_YAMLparams_entry.get()
    volprefix: str = volume_prefix_entry.get()
    regionprefix: str = regions_prefix_entry.get()


    # Labels in the mask files have both names 'strings' and integer-value indices. 'mask_values' is a dict which maps
    # from the label index to its name.
    mask_values: Dict[int, str] = dict_from_file(input_maskvalues_entry.get(), ':')


    # Read in the participant IDs to be operated on. If any of the participants run into an issue, you can simply
    # exclude their ID from the args.participants file 
    participant_IDs: List[str] = lines_from_file(input_IDs_file_entry.get())

    
    # As it says, creates the output folders and subfolders. First checks whether the folders have been created and simply
    # continues if not. Also note in the processing loop below that any output files (L_output and R_output) will
    # also cause the loop to skip that participant. This means that if you want a fresh run with empty folders
    # and files, you will need to either move or delete the outputs of the previous run from your args.output location.
    # To try and ameliorate this, verbose_output_location() adds date and parameter information to the output foldername,
    # so if you are running on a different day and with a different parameter value, then you can simply leave the old
    # run where it is.
    output_location = verbose_output_location(output_location, parameters)
    create_output_folders(output_location, participant_IDs)
    

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * >
    # Start looping over the individual participants and extracting features
    for i, ppt in enumerate(participant_IDs):
        print("\n")
        ROI_found, ROI_path = resolve_target_filepath(regions, regionprefix,  ppt)        # (bool, str)
        T1_found, T1_path   = resolve_target_filepath(volumes, volprefix,   ppt)        # (bool, str)

        # Make sure that each of the required images actually exists and can be found - continue to the next participant if not
        # (I want to check each separately to report to the user, but this could probably be better achieved with a match or something)
        if not ROI_found:
            print(f"      !> Found no Left ROI for participant. '{ppt}' - Continuing...")
            continue
        if not T1_found:
            print(f"      !> Found no Volume for participant. '{ppt}' - Continuing...")
            continue

        # Create the filepaths for the output text files
        feature_outpath: str = os.path.join(output_location, f"{ppt}_Features", f"{ppt}_FeatureValues.txt")


        # If they already exist, then we can skip - mostly this is to avoid having to rerun feature extraction for participants you've already
        # completed if the program hits an error. Additionally, since the program opens the output files in 'append' mode, this also avoids
        # appending duplicate results to the files. As it says, if you want to rerun feature extraction, just move or delete the old files!
        if os.path.exists(feature_outpath):
            print(f"\n  !> Output(s) already found at {feature_outpath}: Continuing...")
            print(f"       (NOTE: If you want to rerun feature extraction, make sure to delete or move the old results first!)\n")
            continue

        print(f"\n!> Operating on ID: {ppt}\nReading T1 from: {T1_path}\nReading ROI from: {ROI_path}\n")
        
        
        # NOTE - The decompression checking and decomp loop + cleanup is unnecessary guff. I might add this back in later if it becomes an issue,
        # but at the moment handling the case that some files might be compressed and others not is unnecessary complexity        
        # Now loop over each of the masks in the ROI file. Loop over the .keys(), not the values, since we need to pass the
        # integer index as an argument to extractor.execute()'s 'label' parameter. The integer index for a given type of segmentation
        # file will always be the same
        for index in mask_values.keys():
            print(f"    => Current Mask: '{mask_values[index]}'\tLabel Index = {index} ({ppt})")

            # Instantiate a new feature extractor
            # Run the feature extraction from this region on the left and right hemispheres
            # There's something off with the mask files, however, so make sure to set correctMask = True in order to have
            # it resliced and fix whatever this geometry mismatch issue is
            # If a given mask is empty, only a single voxel, or does not have the minimum number of dimensions, a ValueError
            # is raised. We don't actually want to crash here, so just print the error, write a $-prefixed line to the output
            # indicating a problem, and continue. The error message itself is also written to the output file so you can check it later
            extractor = featureextractor.RadiomicsFeatureExtractor(parameters, correctMask = True)

            try:
                feature_result = extractor.execute(T1_path, ROI_path, label = index)
            except ValueError as error:
                print("Feature extraction reported an error:", error)
                with open(feature_outpath, 'a') as feat_out:
                    feat_out.write(f"\n$MASKERROR[{mask_values[index]}:{index}] {error}\n")
                continue


            # Feature results are stored in dictionaries. Write each key:value feature pair for a given structure to the output
            # file. Data for each mask in the ROI file will begin with $[MASKNAME:INDEX] on a separate line, to make it easy for
            # my other program to detect mask breaks and split up the data accordingly later on - we need to make spreadsheets :)
            # Each key:value pair is separated only by a colon, again to make it easy to chunk up the data during processing later.
            # It won't really be human-usable, but you're not really supposed to be able to read the raw text dump anyway.
            with open(feature_outpath, 'a') as feat_out:
                feat_out.write(f"\n$[{mask_values[index]}:{index}]")
                for feature, value in six.iteritems(feature_result):
                    feat_out.write(f"\n{feature}:{value}")
                feat_out.write("\nMASKBREAK\n")


            # Clean up local variables for the next iteration
            del(extractor)
            del(feature_result)


        # Print a separator line to clean up the console output
        print(f"\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\n\n")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >
# - - - - - - - - - - - - User Interface Setup & File Selection Functions - - - - - - - - - - - - >
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >

def select_volumes_directory():
    volume_directory = filedialog.askdirectory()
    input_volumes_entry.delete(0, tk.END)
    input_volumes_entry.insert(0, volume_directory)
    


def select_regions_directory():
    regions_directory = filedialog.askdirectory()
    input_regions_entry.delete(0, tk.END)
    input_regions_entry.insert(0, regions_directory)



def select_YAML_file():
    file_types = (("YAML", "*.yml"), ("YAML", "*.yaml"), ("All files", "*.*"))
    filename = filedialog.askopenfilename(title = "Select Parameters (YAML) file...", filetypes = file_types)
    input_YAMLparams_entry.delete(0, tk.END)
    input_YAMLparams_entry.insert(0, filename)



def select_maskvalues_file():
    file_types = (("Text file", "*.txt"), ("All files", "*.*"))
    filename = filedialog.askopenfilename(title = "Select Mask Values file...", filetypes = file_types)
    input_maskvalues_entry.delete(0, tk.END)
    input_maskvalues_entry.insert(0, filename)



def select_ID_file():
    file_types = (("Text file", "*.txt"), ("All files", "*.*"))
    filename = filedialog.askopenfilename(title = "Select Participant ID file...", filetypes = file_types)
    input_IDs_file_entry.delete(0, tk.END)
    input_IDs_file_entry.insert(0, filename)



def check_file_compression():
    checkmessage: str = None

    if files_compressed is True:
        checkmessage = "File compression selected\nVolumes / T1 files will attempt to be decompressed"
    else:
        checkmessage = "No file compression selected\nVolumes / T1 data will be read without decompression"

    tk.messagebox.showinfo(title = "Compression info...", message = checkmessage)



def destroy_window():
    print(f"[{current_time()}] Closing down...")
    parent.destroy()



def run_radiomics():
    print("Time to run!")



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >
# - - - - - - - - - - - - - - ! BZZT, BZZT WARNING - GLOBAL SCOPE ! - - - - - - - - - - - - - - - >
#  Set up the main GUI interface with buttons and text-boxes for entering parameters, files, etc. >
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - >
# Create the main GUI window and resize
parent = tk.Tk()
parent.title("Radiomics!")
parent.geometry("435x580")


# Display a nice 
background_path: str = os.path.join(os.getcwd(), "WindowBackgroundImg.png")

if object_exists(background_path):
    background_image = tk.PhotoImage(file = background_path)
    background_label = tk.Label(parent, image = background_image)
    background_label.place(x = 0, y = 0)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Create button to select the directory containing the T1
# images (or whatever volumes) that radiomics features are to
# be extracted from
input_volumes_directory = tk.Label(parent, text="T1 / Volumes Directory:")
input_volumes_directory.pack()
input_volumes_entry = tk.Entry(parent)
input_volumes_entry.pack()
input_volumes_button = tk.Button(parent, text = "Select", command = select_volumes_directory)
input_volumes_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Create button to select the directory containing the region-
# of-interest (ROI) images that mask regions the user wishes to
# extract radiomics features from
input_regions_label = tk.Label(parent, text = "Region-of-Interest Folder:")
input_regions_label.pack()
input_regions_entry = tk.Entry(parent)
input_regions_entry.pack()
input_regions_button = tk.Button(parent, text = "Select", command = select_regions_directory)
input_regions_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Select the YAML parameters file expected by PyRadiomics.
# For more information on what this should contain, see:
# https://pyradiomics.readthedocs.io/en/latest/customization.html
input_YAMLparams_label = tk.Label(parent, text = "Radiomics YAML Parameter File:")
input_YAMLparams_label.pack()
input_YAMLparams_entry = tk.Entry(parent)
input_YAMLparams_entry.pack()
input_YAMLparams_button = tk.Button(parent, text = "Browse...", command = select_YAML_file)
input_YAMLparams_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Select the text (.txt) file containing the maskname:value
# pairs contained within the ROI files (unfortunately there is
# no real way to do this automatically without information
# about the correct atlas, and the atlas itself).
input_maskvalues_label = tk.Label(parent, text = "ROI Mask Index (Value) File:")
input_maskvalues_label.pack()
input_maskvalues_entry = tk.Entry(parent)
input_maskvalues_entry.pack()
input_maskvalues_button = tk.Button(parent, text = "Browse...", command = select_maskvalues_file)
input_maskvalues_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# 
input_IDs_file_label = tk.Label(parent, text = "Participant ID List File:")
input_IDs_file_label.pack()
input_IDs_file_entry = tk.Entry(parent)
input_IDs_file_entry.pack()
input_IDs_file_button = tk.Button(parent, text = "Browse...", command = select_ID_file)
input_IDs_file_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Check-box allowing user to specify if their files require
# decompression before each step of feature extraction
files_compressed: bool = False
files_compressed_checkbox = tk.Checkbutton(parent, text = "Are files compressed?", variable = files_compressed, onvalue = True, offvalue = False, command = check_file_compression)
files_compressed_checkbox.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Text-entry box which specifies the prefix of T1 / volume
# files (i.e., if there is some text before the ppt. ID)
# Can be left blank if the filenames have no prefix
volume_prefix_label = tk.Label(parent, text = "Volume Filename Prefix:")
volume_prefix_label.pack()
volume_prefix_entry = tk.Entry(parent)
volume_prefix_entry.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Text-entry box which specifies the prefix of the ROI files
# (i.e., if there is some text before the ppt. ID). Again, can
# be left blank if the filenames have no prefix
regions_prefix_label = tk.Label(parent, text = "ROI Filename Prefix:")
regions_prefix_label.pack()
regions_prefix_entry = tk.Entry(parent)
regions_prefix_entry.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Text-entry box to specify the location where the output
# folder will be created. The output folder will be created
# inside this folder automatically.
output_location_label = tk.Label(parent, text = "Output Folder Location:")
output_location_label.pack()
output_location_entry = tk.Entry(parent)
output_location_entry.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Create a button which allows the user to run the start-up
# checks independently of the radiomics feature extraction.
# Allows the user to correct any mistakes before the extraction
# proper begins to run.
run_sanitychecks_button = tk.Button(parent, text = "Run Input Checks", command = perform_startup_checks)
run_sanitychecks_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Create a button to execute the radiomics analysis
run_radiomics_button = tk.Button(parent, text = "Run Radiomics!", command = main)
run_radiomics_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Destroy parent window and quit
quit_button = tk.Button(parent, text = "Quit...", command = destroy_window)
quit_button.pack()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
# Start the GUI main loop
parent.mainloop()