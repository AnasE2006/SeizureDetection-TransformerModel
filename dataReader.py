import os
import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, resample
from scipy.fft import rfft, rfftfreq
import fnmatch

def get_all_TUSZ_2023_session_paths(rootPath):
    session_paths = []
    all_patients = []
    reference_type_count = {}
    all_patients = os.listdir(rootPath)
    for patient in all_patients:
        patient_sessions = os.listdir(os.path.join(rootPath,patient))
        for patient_session in patient_sessions:
            reference_types = os.listdir(os.path.join(rootPath,patient,patient_session))
            for reference_type in reference_types:
                if reference_type not in reference_type_count:
                    reference_type_count[reference_type] = 1
                else:
                    reference_type_count[reference_type] += 1
                files = os.listdir(os.path.join(rootPath,patient,patient_session,reference_type))
                sessions = []
                for file in files:
                    if file.endswith('.edf'):
                        sessions.append(file.split('.')[0])                  
                for session in sessions:
                    session_paths.append(os.path.join(rootPath,patient,patient_session,reference_type,session+'.edf'))
    return session_paths, all_patients, reference_type_count

def get_channels_from_raw(raw):    
    all_25_le = ['EEG FP1-LE', 'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE', 'EEG FP2-LE', 'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG A1-LE', 'EEG T3-LE', 'EEG C3-LE', 
                 'EEG CZ-LE', 'EEG C4-LE', 'EEG T4-LE', 'EEG FP1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG FP2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 
                'EEG O1-LE', 'EEG O2-LE', 'EEG A2-LE']
    
    all_25_REF = ['EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 
                 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG FP1-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 
                'EEG O1-REF', 'EEG O2-REF', 'EEG A2-REF']
    
    ref = False
    for ch in raw.ch_names:
        if fnmatch.fnmatch(ch,"EEG *-LE"):
            print()
            ref = False
            break
        else:
            ref = True
            break
    
    try:    
        if ref and set(all_25_REF).issubset(raw.ch_names):
            signals_1 = raw.get_data(picks=["EEG FP1-REF", "EEG F7-REF", "EEG T3-REF", "EEG T5-REF", "EEG FP2-REF", "EEG F8-REF", "EEG T4-REF", "EEG T6-REF", "EEG A1-REF"]) 
            next_signals_1 = raw.get_data(picks=["EEG T3-REF","EEG C3-REF","EEG CZ-REF","EEG C4-REF","EEG T4-REF","EEG FP1-REF","EEG F3-REF"])
            signals_1 = np.concatenate((signals_1,next_signals_1),axis=0)
            remaining_signals_1 = raw.get_data(picks=['EEG C3-REF', 'EEG P3-REF', 'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF'])
            signals_1 = np.concatenate((signals_1,remaining_signals_1),axis=0)

            signals_2 = raw.get_data(picks=['EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', "EEG O1-REF", "EEG F8-REF", "EEG T4-REF", "EEG T6-REF", "EEG O2-REF"])  
            next_signals_2 = raw.get_data(picks=["EEG T3-REF","EEG C3-REF","EEG CZ-REF","EEG C4-REF","EEG T4-REF","EEG A2-REF","EEG F3-REF"])
            signals_2 = np.concatenate((signals_2,next_signals_2),axis=0)
            remaining_signals_2 = raw.get_data(picks=['EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF',"EEG O2-REF"])
            signals_2 = np.concatenate((signals_2,remaining_signals_2),axis=0)
        elif not ref and set(all_25_le).issubset(raw.ch_names):       
            signals_1 = raw.get_data(picks=["EEG FP1-LE", "EEG F7-LE", "EEG T3-LE", "EEG T5-LE", "EEG FP2-LE", "EEG F8-LE", "EEG T4-LE", "EEG T6-LE", "EEG A1-LE"])  
            next_signals_1 = raw.get_data(picks=["EEG T3-LE","EEG C3-LE","EEG CZ-LE","EEG C4-LE","EEG T4-LE","EEG FP1-LE","EEG F3-LE"])
            signals_1 = np.concatenate((signals_1,next_signals_1),axis=0)
            remaining_signals_1 = raw.get_data(picks=['EEG C3-LE', 'EEG P3-LE', 'EEG FP2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE'])
            signals_1 = np.concatenate((signals_1,remaining_signals_1),axis=0)

            signals_2 = raw.get_data(picks=['EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE', "EEG O1-LE", "EEG F8-LE", "EEG T4-LE", "EEG T6-LE", "EEG O2-LE"])
            next_signals_2 = raw.get_data(picks=["EEG T3-LE","EEG C3-LE","EEG CZ-LE","EEG C4-LE","EEG T4-LE","EEG A2-LE","EEG F3-LE"])
            signals_2 = np.concatenate((signals_2,next_signals_2),axis=0)
            remaining_signals_2 = raw.get_data(picks=['EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE',"EEG O2-LE"])
            signals_2 = np.concatenate((signals_2,remaining_signals_2),axis=0)
    except:
        print('Something is wrong when reading channels of the raw EEG signal')
        flag_wrong = True
        return flag_wrong, 0
    else:
        flag_wrong = False
    
    return flag_wrong, signals_1-signals_2
    
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def slice_signals_into_binary_segments(signals, thisFS, labels, segment_interval, seizure_types, seizure_overlapping_ratio):
    segments = [[] for i in range(len(seizure_types))]
    for this_label in labels:
        if this_label[2] == 'bckg':
            label_index = 0
        else:
            label_index = 1
        seg = []
        for i in range(this_label[0]*thisFS, this_label[1]*thisFS, int(segment_interval*(1-seizure_overlapping_ratio[label_index])*thisFS)):
            
            if i+segment_interval*thisFS > this_label[1]*thisFS:
                break
            
            one_window = []
            noise_flag = False
            incomplete_flag = False
            for j in range(len(signals)):
                this_channel = signals[j][i:i+segment_interval*thisFS]
                if len(this_channel) < segment_interval*thisFS:
                    incomplete_flag = True
                    break
                if max(abs(this_channel)) > 500/10**6:
                    noise_flag = True
                    break
                one_window.append(this_channel)
                
            if incomplete_flag==False and noise_flag==False and one_window and len(one_window[0]) == thisFS*segment_interval:
                seg.append(np.array(one_window))
        segments[label_index].append(seg)
    return segments  

def slice_signals_into_multiclass_segments(signals, thisFS, labels, segment_interval, seizure_types, seizure_overlapping_ratio):
    segments = [[] for i in range(len(seizure_types))]

    for this_label in labels:
        if this_label[2] not in seizure_types:
            print('Seizure type not included: ', this_label[2])
            continue
        label_index = seizure_types.index(this_label[2])
        seg = []
        for i in range(this_label[0]*thisFS, this_label[1]*thisFS, int(segment_interval*(1-seizure_overlapping_ratio[label_index])*thisFS)):
            
            if i+segment_interval*thisFS > this_label[1]*thisFS:
                break
            
            one_window = []
            noise_flag = False
            incomplete_flag = False
            for j in range(len(signals)):
                this_channel = signals[j][i:i+segment_interval*thisFS]
                if len(this_channel) < segment_interval*thisFS:
                    incomplete_flag = True
                    break
                if max(abs(this_channel)) > 500/10**6:
                    noise_flag = True
                    break
                one_window.append(this_channel)
            if incomplete_flag==False and noise_flag==False and one_window and len(one_window[0]) == thisFS*segment_interval:
                seg.append(np.array(one_window))
        segments[label_index].append(seg)
    return segments  
 
def plot_signal_in_frequency(signal, filtered_signal, sample_rate):
    # Compute the frequency representation of the signals
    fft_orig = rfft(signal)
    fft_filtered = rfft(filtered_signal)

    # Compute the frequencies corresponding to the FFT output elements
    freqs = rfftfreq(len(signal), 1/sample_rate)

    # Plot the original signal in frequency domain
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(freqs, np.abs(fft_orig))
    plt.title('Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Plot the filtered signal in frequency domain
    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(fft_filtered))
    plt.title('Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

def make_a_filtered_plot_for_comparison(signals, filtered_signals, thisFS):
    plt.figure()
    plt.clf()
    maximum_samples = 200
    channel_index = 5
    if maximum_samples == -1:
        t = np.linspace(0, signals.shape[1]/thisFS, signals.shape[1])
        plt.plot(t, signals[channel_index,:], label='Noisy signal')
        plt.plot(t, filtered_signals[channel_index][:], label='Filtered signal')
    else: 
        t = np.linspace(0, maximum_samples/thisFS, maximum_samples)    
        plt.plot(t, signals[channel_index,:maximum_samples], label='Noisy signal')
        plt.plot(t, filtered_signals[channel_index][:maximum_samples], label='Filtered signal')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.savefig('filtered_signal_plot.png')

def resample_data_in_each_channel(signals, thisFS, resampleFS):
    sigResampled = []
    for sig in signals:
        if type(sig) == np.ndarray:
            num = int(sig.shape[0]/thisFS*resampleFS)
        else:
            num = int(len(sig)/thisFS*resampleFS)
        y = resample(sig, num)
        sigResampled.append(y)
    return sigResampled