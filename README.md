# Instructions for Running `inference.py` File

1. Create a new conda environment using the given command:  
   `conda create --name <environment_name> python=3.9`

2. Activate the environment using the following command:  
   `conda activate <environment_name>`

3. Install the required dependencies using the `requirements.txt` file with the following command:  
   `pip install -r requirements.txt`

4. Keep the `weights` folder and the `inference.py` file in the same parent directory before running the `inference.py` file.

5. Run the `inference.py` file using the following command:  
   `python3 inference.py --image_dir <path_to_image_directory> --csv_path <path_to_metadatacsv_file>`
