IMPORTANT: Kindly update path, path_data variables in settings.py to the correct paths of the output and data folders respectively (with the path of directory in which you save this code).

The training file is large and hence, cannot be uploaded on Git. Please download the file using the following link:
https://drive.google.com/file/d/0B_gDCeR3Cp-3Z0p0MXBZeXNuZHM/view?usp=sharing

Paste this file in the data folder with the same name (training_datav2.txt).

Run the files in the following order:

1) raw_data_to_pre_proc.py
2) pre_proc_to_corpora.py
3) corpora_to_lda.py
4) hellinger_distances.py (optional)
5) lda_to_recommendations.py  --> saves a file bow_u_40.txt in result folder