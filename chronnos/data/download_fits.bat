@echo off
python download.py --path ..\..\..\pipeline\fits_files\download_test --dates "2017-01-01T00:00:00" --hmi_series "hmi.M_720s" --n_workers 8 --resolution 512