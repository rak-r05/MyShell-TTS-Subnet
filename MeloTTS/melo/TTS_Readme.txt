Readme for training (fine-tuning) the MeloTTS model
Writen By: Raksha 
Date: Apr 9, 2024

The modules were downloaded individually and not using the 'pip install -e .' since there were some errors during the execution of the command

Data Preparation
Created a folder for data (<data_folder>). This will contain the data pre-processing script and this is where the prepared data will be stored.
In that folder, create a file named metadata.list and a sub-folder to store the prepared data called <data_folder>/prepared_data

The config.json was changed to make num_lang as 10 in both the fields before training

