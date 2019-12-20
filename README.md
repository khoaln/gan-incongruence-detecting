This documentation is to help reproduce the experiments taken in the DAT620 project "Generative adversarial networks for detecting Incongruence between News Headline and Body Text"

## Requirements
* Python3
* Tensorflow >= 1.4 (tested on Tensorflow 1.4.1)
* numpy
* tqdm
* sklearn
* nltk
* sentence-transformers
* gensim


Please remember to update the parameters (such as the directories) to the correct values on your system for all of the following commands.
## Generate headlines
The GAN model should be run on a GPU server

* Dataset

    Please follow the instructions [here](https://github.com/abisee/cnn-dailymail) for downloading and preprocessing the CNN/DailyMail dataset. After that, store data files ```train.bin```, ```val.bin``` and vocabulary file ```vocab``` into specified data directory, e.g., ```./data/```.
    Download the NELA-17 dataset from [here](https://github.com/BenjaminDHorne/NELA2017-Dataset-v1) and apply the same way of pre-processing to get the ```test.bin``` file which we want to generate the headlines for.

* Prepare negative samples for discriminator

    You can download the prepared data ```discriminator_train_data.npz``` for discriminator from [dropbox](https://www.dropbox.com/s/i1otqkrsgup63pt/discriminator_train_data.npz?dl=0) and store into specified data directory, e.g., ```./data/```.

* Pre-train the generator

    ```
    python3 main.py --mode=pretrain --data_path=../dataset/data_generation_for_gan/finished_files/train.bin --vocab_path=../dataset/data_generation_for_gan/finished_files/vocab --log_root=./log/ --restore_best_model=False
    ```
    
* Train the full model

    ```
    python3 main.py --mode=train --data_path=../dataset/data_generation_for_gan/finished_files/train.bin --vocab_path=../dataset/data_generation_for_gan/finished_files/vocab --log_root=./log/ --pretrain_dis_data_path=./data/discriminator_train_data.npz --restore_best_model=False --vocab_size=50000
    ```
   
* Decode

    ```
    python3 main.py --mode=decode --data_path=../dataset/data_generation_for_gan/finished_files/test.bin --vocab_path=../dataset/data_generation_for_gan/finished_files/vocab --log_root=./log/ --single_pass=True
    ```
    
* Follow [here](https://github.com/iwangjian/textsum-gan) for more details
    
## Calculate similarity scores

* Combine the generated headlines, original headlines and body text to prepare dataset for prediction model

    ```
    python3 dataset_creation.py --mode=export --generated_data_path=
    ```
* Calculate similarity scores from BERT embeddings

    ```
    python3 similarity_calculation_Bert.py --data_path= --output_path=
    ```
    
* Calculate similarity scores from Doc2vec embeddings

    ```
    python3 similarity_calculation_Doc2Vec.py --data_path= --output_path=
    ```
    
## Run classification

* SVM + BERT/Doc2Vec (please modify the correct path to the files inside the code first):

    ```
    python3 classifier.py
    ```
    
* Similarly for UCLMR + BERT/Doc2Vec:

    ```
    python3 nn_classifier.py
    ```
 
