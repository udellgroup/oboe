This is a quick example to show how to generate the error matrix from preprocessed datasets. It works on Unix and Linux but not on OS X for now.

# Dataset format

The datasets should be `csv` files. All the columns except the last are features; the last column is the class label.

# Recording model errors and runtime
Run
```
bash generate.sh
```
It will create a `results` directory, with a subdirectory named by the start time of the generation procedure and containing results on individual datasets. We call this subdirectory the "csv directory".
# Merging into the error and runtime matrices
First, modify the directory name in angle brackets in `merge.sh`to be the name of the "csv directory". Then do
```
bash merge.sh
```
It will generate a `error_matrix.csv` and a `runtime_matrix.csv` in the "csv directory", and move the csv files already merged into these matrices into `merged_csv_files`.
