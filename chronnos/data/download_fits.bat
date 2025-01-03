@echo off
python download.py --path ..\..\..\pipeline\fits_files\paper_examples --dates "2024-09-20T00:00:00" "2024-09-21T00:00:00" --hmi_series "hmi.M_720s" --n_workers 10

@REM --resolution 512


