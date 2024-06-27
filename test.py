from audioset_download import Downloader

d = Downloader(root_path='audioset', labels=None, n_jobs=128, download_type='unbalanced_train', copy_and_replicate=False)
d.download(format="best", quality="best")
