import os
import numpy as np
import mne
from dataReader import *
import random
import shutil
from scipy.signal import iirnotch, lfilter

# Defining vars to be used later
low = 0.5
high = 120
samplFreq = 1024
resampleFS = 250
seizure_types = ['bckg', 'seizure']
seizure_session_downsampling_ratio = [1, 1]
seizure_overlapping_ratio = [0, 0.75]
segment_interval = 4
data_mode = 'full'

# Create the notch filter
notch_1_b, notch_1_a = iirnotch(1, Q=30, fs=resampleFS)
notch_60_b, notch_60_a = iirnotch(60, Q=30, fs=resampleFS)

# Set up the proper paths for the dataset and where it will be saved (*Edit for your file path*)
train_root = os.path.join("/xxx","SeizureData","edf","train") # Path to the training and validation data
test_root = os.path.join("/xxx","SeizureData","edf","dev") #Path to the testing data
preprocessed_root = os.path.join("/xxx","SeizureData","PreProcessedData") # Path to where preprocessed data will be savwd
if not os.path.exists(preprocessed_root):
    os.mkdir(preprocessed_root)


# Delete any previously preprocessed data 
if not os.path.exists(os.path.join(preprocessed_root, 'segment_interval_'+str(segment_interval)+'_sec')):
    os.mkdir(os.path.join(preprocessed_root, 'segment_interval_'+str(segment_interval)+'_sec'))
else:
    for filename in os.listdir(os.path.join(preprocessed_root, 'segment_interval_'+str(segment_interval)+'_sec')):
        file_path = os.path.join(preprocessed_root, 'segment_interval_'+str(segment_interval)+'_sec', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


# Extract EEG session data
train_paths, train_patients, train_reference_type_count = get_all_TUSZ_2023_session_paths(train_root)
test_paths, test_patients, test_reference_type_count = get_all_TUSZ_2023_session_paths(test_root)
total_paths = train_paths + test_paths

training_data = [[] for i in range(len(seizure_types))]
validation_data = [[] for i in range(len(seizure_types))]
testing_data = [[] for i in range(len(seizure_types))]

# 1. Separate patients into two groups: with seizures and without seizures
patients_with_seizures = []
patients_without_seizures = []
for patient in train_patients:
	for session in os.listdir(train_root + "/" + patient):
		for folder in os.listdir(train_root + "/" + patient + "/" + session):
			for file in os.listdir(train_root + "/" + patient + "/" + session + "/" + folder):
				if file.endswith(".csv_bi"):
					hasSeizure = False
					with open(train_root + "/" + patient + "/" + session + "/" + folder + "/" + file, 'r') as file:
						lines = file.readlines() 
						for line in reversed(lines):   
							if line.split(",")[0] == "channel":
								break                 
							else:
								classification = line.split(",")[3]
								if classification == "seiz":
									hasSeizure = True
	
	if hasSeizure:
		patients_with_seizures.append(patient)
	else:
		patients_without_seizures.append(patient)
		
# 2. Shuffle the lists to randomize the order
random.seed(42) 
random.shuffle(patients_with_seizures)
random.shuffle(patients_without_seizures)

# 3. Manually split the patients into 80% for training and 20% for validation
train_size_seizures = int(0.8 * len(patients_with_seizures))
train_size_bckg = int(0.8 * len(patients_without_seizures))

# Split the seizure patients (80% for training, 20% for validation)
train_patients_with_seizures = patients_with_seizures[:train_size_seizures]
val_patients_with_seizures = patients_with_seizures[train_size_seizures:]

# Split the background patients (80% for training, 20% for validation)
train_patients_without_seizures = patients_without_seizures[:train_size_bckg]
val_patients_without_seizures = patients_without_seizures[train_size_bckg:]

# 4. Combine the seizure and background patients for both training and validation sets
train_patients = train_patients_with_seizures + train_patients_without_seizures
val_patients = val_patients_with_seizures + val_patients_without_seizures


# Iterate through all the paths & patients 
count_session = 0
for data_path in total_paths:
	if 'train' in data_path:
		patient = data_path.split('train/')[1].split('/')[0]
		reference_type = data_path.split('train/')[1].split('/')[2]
		if reference_type == "03_tcp_ar_a":
			continue
		patient_session = data_path.split('train/')[1].split('/')[-1][:-4]
	elif 'dev' in data_path:
		patient = data_path.split('dev/')[1].split('/')[0]
		reference_type = data_path.split('dev/')[1].split('/')[2]
		if reference_type == "03_tcp_ar_a":
			continue
		patient_session = data_path.split('dev/')[1].split('/')[-1][:-4]
	else:
		continue
		
	if patient in train_patients:
		flag_train_val_test = 'train'
	elif patient in val_patients:
		flag_train_val_test = 'val'
	elif patient in test_patients:
		flag_train_val_test = 'dev'
	
    # Read EEG data from EDF file and extract the signals
	count_session += 1
	raw = mne.io.read_raw_edf(data_path, verbose='warning')
	thisFS = int(raw.info['sfreq'])
	if thisFS != 250:
		continue	
	flag_wrong, signals = get_channels_from_raw(raw)
	if flag_wrong:
		continue
	
    #Filter EEG signals using vars defined earlier and resample
	filtered_signals = []
	for i in range(signals.shape[0]):
		bandpass_filtered_signal = butter_bandpass_filter(signals[i,:], low, high, samplFreq, order=3)
		filtered_1_signal = lfilter(notch_1_b, notch_1_a, bandpass_filtered_signal)
		filtered_60_signal = lfilter(notch_60_b, notch_60_a, filtered_1_signal)
		filtered_signals.append(filtered_60_signal)
	resampled_signals = []
	if thisFS == resampleFS:
		resampled_signals = filtered_signals[:]
	else:
		resampled_signals = resample_data_in_each_channel(filtered_signals, thisFS, resampleFS)
		

    # Read the seiz/bckg labels from binary file and assign them to signals properly
	labels = []
	tseFile = data_path[:-4] + '.csv_bi'
	with open(tseFile,'r') as tseReader:
		rawText = tseReader.readlines()[6:]
		seizPeriods = []
		for item in rawText:
			labels.append([int(item.split(",")[1].split('.')[0]),int(item.split(",")[2].split('.')[0]),item.split(",")[3]])
	segments = slice_signals_into_binary_segments(filtered_signals, thisFS, labels, segment_interval, seizure_types, seizure_overlapping_ratio)
	
    
	for i in range(len(segments)):
		if segments[i] and segments[i][0]:
			this_array  = []
			this_labels = seizure_types[i]
			for j in range(len(segments[i])):
				if not segments[i][j]:
					continue
				for k in range(len(segments[i][j])):
					this_array.append(segments[i][j][k])
				
			save_dir = os.path.join(preprocessed_root, 'segment_interval_' + str(segment_interval) + '_sec', flag_train_val_test, this_labels)
			os.makedirs(save_dir, exist_ok=True)
			save_file = os.path.join(save_dir, patient_session+'.npy')
			if os.path.isfile(save_file):
				existing_data = np.load(save_file, allow_pickle=True)
				new_data = np.concatenate((existing_data, np.array(this_array)))
				np.save(save_file, new_data)
				print(new_data.shape)
			else:
				print(np.array(this_array).shape)
				np.save(save_file, np.array(this_array))
	if data_mode == 'small':
		if count_session >= 1500:
			break
	elif data_mode == 'tiny':
		if count_session >= 100:
			break
	elif data_mode == 'large':
		if count_session >= 2500:
			break
