## Datasets

This project uses several datasets for training and evaluation. Please ensure you have the following datasets available:

- huggingface datasets: `taewan/kitti_nyu`

You can download the datasets from the following links:

- [taewan/kitti_nyu](https://huggingface.co/datasets/taewan2002/kitti_nyu)

```bash
git lfs install
git clone https://huggingface.co/datasets/taewan2002/kitti_nyu
cd kitti_nyu
```

```bash
for f in *.tar.gz; do tar -xzvf "$f" && rm "$f"; done
mv 2011_09_26_2/* 2011_09_26_1/
rmdir 2011_09_26_2
mv 2011_09_26_1 2011_09_26
```

Make sure to place the datasets in the appropriate directory before running the project.