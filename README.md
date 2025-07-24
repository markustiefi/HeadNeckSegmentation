# HeadNeckSegmentation
 
<iframe src="./documentation/graphical_abstract.pdf" width="100%" height="600px"></iframe>

conda create --name headneck python=3.9.18
conda activate headneck
pip install -r requirements.txt

Python version 3.9.18

To run the inference which takes ".nii.gz" files and returns the segmentation mask: use 
python segmentation_pipeline_inference.py

To run the training for the global seed point extractor run
python main_global.py

To run the training for the local network run
python main_local.py

Link to network weights:
https://fileshare.uibk.ac.at/d/ca920a9994d34cd5b00c/

Link to data:
https://fileshare.uibk.ac.at/d/6d92002cc58741698522/
