#  MAML-Pytorch

This is an extension of [MAML-Pytorch by Liangqu Long](https://github.com/dragen1860/MAML-Pytorch).
<br>This version allows easy use of a custom dataset, and offers an option for "further training".
<br>The further training step is typically referred to as fine-tuning in machine-learning. <br>However, this term is already used in MAML to 
describe the step where the query set is evaluated.

Logits and predictions are exported to CSV files (All_logits.csv and test_predictions_and_labels, respectively). 
<br>Test accuracy (mean values for analysis of the query set) are output to mean_test_accuracy.txt .
<br>Various metrics are also calculated and output to mean_metrics.csv and metrics_summary.csv . 
<br>mean_metrics.csv includes training and validation (i.e., support and query) loss and accuracy values for each epoch.
<br>The summary file includes mean support loss, mean test/query loss, test AUC, and test F1 scores (macro and micro).

The code has been tested using Google Colab. <br>It requires Python 3.x, PyTorch 0.4+ and CUDA-powered GPUs.



## Data Requirements
    
The training, validation and testing, directories are expected to contain all relevant images,
<br>along with a CSV file listing the names of all relevant files. This can be generated using os.listdir() .

For this version, each training directory (i.e, including the further training directory) must contain a file named train.csv. 
<br>If further training is being used, each training dataset is expected to be unique.
<br> The validation set (query set/fine-tuning set) and the testing set are generated from the target dataset.

Below is an example of the contents of train.csv:

     	filename	            label
    0	sample0001.png	      group1
    1	sample0002.png	      group2
    3	sample0003.png	      group1
    ...


## Usage

In a Jupyter/Colab notebook:


Retrieve the project files 

    !git clone https://github.com/MTynes/MAML-Pytorch.git maml_pytorch
	


Set the parameters 

	train_directory ='/content/miniimagenet/images'
	validation_directory ='/content/all_validation_images'
	test_directory ='/content/all_test_images'
	further_training_directory = '/content/all_further_training_images'

	n_epochs = 200 * 10000 # As with the original implementation, this value must be a multiple of 10000
	ft_n_epochs = 200 * 10000


Run the training file

    !python /content/maml_pytorch/train_custom_dataset.py --run_further_training 'true' --epochs {n_epochs} --train_dir {train_directory} --validation_dir {validation_directory} --test_dir {test_directory} --further_training_dir {further_training_directory} --further_training_epochs {ft_n_epochs}


Examples of use with spectrogram images generated from EEG data are available here:
<br>https://github.com/WinAIML/schizophrenia/blob/master/MLModels/Meta%20Learning%20Models/MAML_Pytorch_with_Dataset-1.ipynb
<br>https://github.com/WinAIML/schizophrenia/blob/master/MLModels/Meta%20Learning%20Models/MAML_Pytorch_with_further_training_Test_Dataset-1.ipynb

Read the help summary for further details

    !python /content/maml_pytorch/train_custom_dataset.py --help


        usage: train_custom_dataset.py [-h] [--train_dir TRAIN_DIR]
									   [--further_training_dir FURTHER_TRAINING_DIR]
									   [--validation_dir VALIDATION_DIR]
									   [--test_dir TEST_DIR]
									   [--run_further_training RUN_FURTHER_TRAINING]
									   [--epochs EPOCHS]
									   [--further_training_epochs FURTHER_TRAINING_EPOCHS]
									   [--n_way N_WAY] [--k_spt K_SPT] [--k_qry K_QRY]
									   [--imgsz IMGSZ] [--imgc IMGC]
									   [--task_num TASK_NUM] [--meta_lr META_LR]
									   [--update_lr UPDATE_LR]
									   [--update_step UPDATE_STEP]
									   [--update_step_test UPDATE_STEP_TEST]
									   [--accuracy_log_file ACCURACY_LOG_FILE]

		optional arguments:
		  -h, --help            show this help message and exit
		  --train_dir TRAIN_DIR
								train data directory
		  --fine_tune_dir FINE_TUNE_DIR
								fine tuning data directory
		  --validation_dir VALIDATION_DIR
								validation data directory
		  --test_dir TEST_DIR   test data directory
		  --run_fine_tuning RUN_FINE_TUNING
								Boolean for adding a second dataset for further
								training. Set as string. Case insensitive.
		  --epochs EPOCHS       Number of epochs
		  --fine_tuning_epochs FINE_TUNING_EPOCHS
								Number of epochs for fine tuning cycle
		  --n_way N_WAY         n way
		  --k_spt K_SPT         k shot for support set
		  --k_qry K_QRY         k shot for query set
		  --imgsz IMGSZ         imgsz
		  --imgc IMGC           imgc
		  --task_num TASK_NUM   meta batch size, namely task num
		  --meta_lr META_LR     meta-level outer learning rate
		  --update_lr UPDATE_LR
								task-level inner update learning rate
		  --update_step UPDATE_STEP
								task-level inner update steps
		  --update_step_test UPDATE_STEP_TEST
								update steps for finetunning
		  --accuracy_log_file ACCURACY_LOG_FILE
								Output file for mean test accuracy


Please refer to the original MAML paper here: [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).

