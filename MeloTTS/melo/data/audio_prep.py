from datasets import load_dataset, concatenate_datasets, Dataset
import librosa
import soundfile as sf
import os
import pandas as pd

dataset = load_dataset("vctk", trust_remote_code=True)['train']
wav_folder_name = 'prepared_data_44k'
resample_freq = 44100

print('dataset load success, Features of the dataset are :\n', dataset.features)
print('----------------------------------------\n')

def filter_speaker(audio_dict, speaker_id):
    return audio_dict['speaker_id']==speaker_id

print('Removing duplcates....')
filtered_datasets = []
for each_sp in list(set(dataset['speaker_id']))[:3]:
    #data1 = dataset.filter(lambda each_per:each_per['speaker_id']==each_sp, num_proc=8)
    data1 = dataset.filter(lambda speaker:speaker['speaker_id']==each_sp, num_proc=14)
    #data2 = data1.select(list(set(data1['text_id'])))
    data2 = pd.DataFrame(data1)
    data2 = data2.drop_duplicates(subset=['text_id'])
    data2 = Dataset.from_pandas(data2)
    '''
    print(data2[:5])
    exit()
    for each_text in list(set(data1['text_id'])):
        data2 = data1.filter(lambda text:text['text_id']==each_text, num_proc=8)[:1]
        print(data2[:5])
        exit()

    print(data1[:10]['text_id'])
    exit()
    '''
    filtered_datasets.append(data2)

prepared_data = concatenate_datasets(filtered_datasets)

out_file = open('metadata.list', 'w')
for inum, each_audio in enumerate(prepared_data):
    print('Processing audio: ', inum, len(prepared_data), end='\r')
    npAudioData, fSr = sf.read(each_audio['file'])
    #print('------------')
    #print(npAudioData, fSr)
    y_44k = librosa.resample(npAudioData, orig_sr=fSr, target_sr=resample_freq)
    #print(npAudioData.shape, y_44k.shape)
    
    sWavFilename = each_audio['text_id'] + each_audio['speaker_id'] + each_audio['file'].split('/')[-1].split('.')[0].split('_')[-1] + '.wav'
    sWavFolderName = os.path.join(os.getcwd(), wav_folder_name)
    sWavFilename = os.path.join(sWavFolderName, sWavFilename)
    sf.write(sWavFilename, y_44k, resample_freq)
    out_file.writelines(sWavFilename+'|EN-US|EN|'+each_audio['text']+'\n')
    
out_file.close()
