##############################################################################################
# Below is the shell command including my hf_token, path to the quora_dtaset and the target directory where the model will be saved
##############################################################################################

python3 train_jihyun.py --hf_token {YOUR_HF_TOKEN_HERE} --data_path jbrophy123/quora_dataset --lora_dir jihyunkim423/quora_hw_dataset
