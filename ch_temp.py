import os

# Create txt file needed for test_val_dataset.py
# CH Added: temp file, needing for training

# txt_dir = "./training_data/"
txt_dir = "./test_data/"
data_dir = os.path.join(txt_dir, "hazy/")
# data_dir = txt_dir

write_file = open(os.path.join(txt_dir, "test.txt"), "w")
for filename in os.listdir(data_dir):
    write_file.write(f"{filename}\n")
write_file.close()
print("done")

