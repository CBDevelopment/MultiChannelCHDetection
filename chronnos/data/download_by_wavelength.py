import argparse
import logging
import os
import asyncio
import aiohttp
import traceback
from datetime import timedelta
from pathlib import Path
from io import BytesIO

import drms
import numpy as np
import pandas as pd
from astropy.io.fits import HDUList, PrimaryHDU, Header
from astropy.io.fits.hdu.compressed import CompImageHDU
from dateutil.parser import parse
from sunpy.map import Map

from chronnos.data.convert import prepMap

from datetime import datetime, timedelta
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class SingleWavelengthFetcher:
    def __init__(self, ds_path, wavelength, num_worker_threads=4, hmi_series='hmi.M_720s', resolution=None, max_retries=2):
        self.ds_path = ds_path
        self.wavelength = str(wavelength)  # Convert to string for consistency
        self.resolution = resolution
        self.max_retries = max_retries
        self.num_worker_threads = num_worker_threads
        self.hmi_series = hmi_series
        
        # Create directory for the specific wavelength
        os.makedirs(ds_path, exist_ok=True)
        os.makedirs(os.path.join(ds_path, self.wavelength), exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f"{ds_path}/wavelength_{wavelength}_log.log"),
                logging.StreamHandler()
            ])
        
        self.drms_client = drms.Client()
        self.download_queue = asyncio.Queue()
        self.session = None

    async def download_worker(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        while True:
            try:
                task = await self.download_queue.get()
                if task is None:
                    self.download_queue.task_done()
                    break

                header, segment, t = task
                await self.process_download(header, segment, t)
                self.download_queue.task_done()
                
            except Exception as e:
                logging.error(f"Worker error: {str(e)}")
                self.download_queue.task_done()

    async def process_download(self, header, segment, t):
        wavelength = str(header.get('WAVELNTH', '6173'))  # Default to HMI wavelength if not found
        dir = os.path.join(self.ds_path, str(wavelength))
        obs_time = header['DATE__OBS']
        formatted_time = pd.to_datetime(obs_time).strftime('%Y-%m-%dT%H%M%S')
        map_path = os.path.join(dir, f"{formatted_time}.fits")
        
        logging.info(f"Processing {wavelength}A data for {formatted_time}")

        if os.path.exists(map_path):
            logging.info(f'File exists: {map_path}, skipping.')
            return

        for attempt in range(self.max_retries):
            try:
                # Ensure proper URL construction with forward slash
                url = 'http://jsoc.stanford.edu' + ('' if segment.startswith('/') else '/') + segment
                logging.info(f"Downloading from JSOC: {wavelength}A")
                async with self.session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status}")
                    fits_data = await response.read()
                    logging.info(f"Download complete for {wavelength}A")
                
                hdul = HDUList.fromstring(fits_data)
                hdul.verify('silentfix')
                data = hdul[1].data
                header = {k: v for k, v in header.items() if not pd.isna(v)}
                header['DATE_OBS'] = header['DATE__OBS']
                
                s_map = Map(data, header)
                if self.resolution:
                    s_map = prepMap(s_map, self.resolution)
                    
                if os.path.exists(map_path):
                    os.remove(map_path)
                    
                compressed_fits = CompImageHDU(
                    data=s_map.data, 
                    header=Header(s_map.meta),
                    compression_type='HCOMPRESS_1',
                    quantize_level=16.0)
                compressed_hdul = HDUList([PrimaryHDU(), compressed_fits])
                compressed_hdul.writeto(map_path)
                
                logging.info(f'Downloaded: {obs_time} / {wavelength}A')
                break
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logging.warning(f'Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {str(e)}')
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f'Failed after {self.max_retries} attempts: {str(e)}')
                    raise

    async def fetchDates(self, dates):
        connector = aiohttp.TCPConnector(limit=self.num_worker_threads, force_close=True)
        self.session = aiohttp.ClientSession(connector=connector)
        
        workers = [asyncio.create_task(self.download_worker()) 
                  for _ in range(self.num_worker_threads)]
        
        try:
            for date in dates:
                formatted_time = date.strftime('%Y-%m-%dT%H%M%S')
                file_path = os.path.join(self.ds_path, self.wavelength, f"{formatted_time}.fits")
                
                if os.path.exists(file_path):
                    logging.info(f"Skipping {date.isoformat()}, data exists.")
                    continue
                    
                try:
                    await self.fetchData(date)
                except Exception as ex:
                    logging.error(f"Error fetching {date.isoformat()}: {traceback.format_exc()}")

            # Signal workers to finish
            for _ in range(self.num_worker_threads):
                await self.download_queue.put(None)

            await self.download_queue.join()
            await asyncio.gather(*workers)
            
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def fetchData(self, time):
        logging.info(f"\nFetching {self.wavelength}A data for timestamp: {time.isoformat()}")
        time_param = f"{time.isoformat('_', timespec='seconds')}Z"
        
        if self.wavelength == '6173':  # HMI magnetogram
            ds = f"{self.hmi_series}[{time_param}]{{magnetogram}}"
            keys = self.drms_client.keys(ds)
            header, segment = self.drms_client.query(ds, key=','.join(keys), seg='magnetogram')
            
            if len(header) != 1 or np.any(header.QUALITY != 0):
                logging.error(f'HMI data not valid for {time.isoformat()}. Using fallback.')
                await self.fetchDataFallback(time)
                return
                
            await self.download_queue.put((header.iloc[0].to_dict(), segment.iloc[0].magnetogram, time))
            
        else:  # AIA EUV
            ds = f'aia.lev1_euv_12s[{time_param}][{self.wavelength}]{{image}}'
            keys = self.drms_client.keys(ds)
            header, segment = self.drms_client.query(ds, key=','.join(keys), seg='image')
            
            if len(header) != 1 or np.any(header.QUALITY != 0):
                logging.error(f'EUV data not valid for {time.isoformat()}. Using fallback.')
                await self.fetchDataFallback(time)
                return
                
            await self.download_queue.put((header.iloc[0].to_dict(), segment.iloc[0].image, time))

    async def fetchDataFallback(self, time):
        logging.info(f"\nUsing fallback method for {time.isoformat()}")
        t = time - timedelta(minutes=1)
        
        if self.wavelength == '6173':  # HMI magnetogram
            ds = f'hmi.M_720s[{t.replace(tzinfo=None).isoformat("_", timespec="seconds")}Z/12h@720s]{{magnetogram}}'
            seg_type = 'magnetogram'
        else:  # AIA EUV
            ds = f'aia.lev1_euv_12s[{t.replace(tzinfo=None).isoformat("_", timespec="seconds")}Z/12h@12s][{self.wavelength}]{{image}}'
            seg_type = 'image'
            
        logging.info(f"Querying with dataset: {ds}")
        
        keys = self.drms_client.keys(ds)
        header_tmp, segment_tmp = self.drms_client.query(ds, key=','.join(keys), seg=seg_type)
        
        if len(header_tmp) == 0:
            raise ValueError('No data found!')
        
        # Process data efficiently
        date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')
        date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - time).abs()
        
        # Create DataFrames with all data
        header_df = pd.DataFrame({
            **{col: header_tmp[col] for col in header_tmp.columns},
            'date_diff': date_diff
        })
        
        segment_df = pd.DataFrame({
            **{col: segment_tmp[col] for col in segment_tmp.columns},
            'date_diff': date_diff
        })
        
        # Sort and filter
        header_df = header_df.sort_values('date_diff')
        segment_df = segment_df.sort_values('date_diff')
        
        quality_mask = header_df.QUALITY == 0
        header_df = header_df[quality_mask]
        segment_df = segment_df[quality_mask]
        
        if header_df.empty:
            raise ValueError("No valid data found after filtering.")

        logging.info(f"Found valid data with time difference: {header_df.iloc[0]['date_diff']}")
        
        header = header_df.iloc[0].drop('date_diff')
        segment = segment_df.iloc[0].drop('date_diff')
        
        await self.download_queue.put((
            header.to_dict(),
            getattr(segment, seg_type),
            time
        ))

async def download_multiple_wavelengths(base_path, wavelengths, dates, hmi_series='hmi.M_720s', 
                                n_workers=4, resolution=None):
    """
    Helper function to download multiple wavelengths for multiple timestamps.
    
    Args:
        base_path (str): Base directory to store downloads
        wavelengths (list): List of wavelengths to download (as strings, e.g. ['171', '193', '6173'])
        dates (list): List of datetime objects for timestamps to download
        hmi_series (str): HMI series to use (default: 'hmi.M_720s')
        n_workers (int): Number of worker threads per wavelength (default: 4)
        resolution (int): Optional resolution to resize images to
    """
    import time as time_mod
    
    total_start_time = time_mod.time()
    logging.info(f"Starting batch download for wavelengths: {wavelengths}")
    logging.info(f"Timestamps to process: {len(dates)}")
    
    for wavelength in wavelengths:
        wavelength_start_time = time_mod.time()
        logging.info(f"\nProcessing wavelength {wavelength}Å")
        
        fetcher = SingleWavelengthFetcher(
            ds_path=base_path,
            wavelength=wavelength,
            hmi_series=hmi_series,
            num_worker_threads=n_workers,
            resolution=resolution
        )
        
        try:
            await fetcher.fetchDates(dates)
        except Exception as e:
            logging.error(f"Error processing wavelength {wavelength}: {str(e)}")
            continue
            
        wavelength_end_time = time_mod.time()
        elapsed = wavelength_end_time - wavelength_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        logging.info(f"Completed wavelength {wavelength}Å in {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    total_end_time = time_mod.time()
    total_elapsed = total_end_time - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    logging.info(f"\nTotal processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logging.info("Batch download complete")

async def parse_analysis_file(file_path: str) -> Tuple[Dict[str, List[datetime]], Dict[str, List[Tuple[datetime, Set[str]]]]]:
    """
    Parse the wavelength analysis file to extract missing dates and wavelengths.
    
    Returns:
        Tuple containing:
        - Dictionary of completely missing dates by year
        - Dictionary of partially missing wavelengths by year
    """
    completely_missing_dates = defaultdict(list)
    partially_missing_dates = defaultdict(list)
    
    current_year = None
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Split content by year sections
    year_sections = re.split(r'={10,}', content)
    
    for section in year_sections:
        if not section.strip():
            continue
            
        # Extract year
        year_match = re.search(r'(\d{4}):', section)
        if not year_match:
            continue
        current_year = year_match.group(1)
        
        # Find completely missing dates
        missing_dates_section = re.search(r'SWPC dates with no wavelength data.*?\n(.*?)\n\n', 
                                        section, re.DOTALL)
        if missing_dates_section:
            for line in missing_dates_section.group(1).strip().split('\n'):
                if 'SWPC Drawing Timestamp:' in line:
                    timestamp_match = re.search(r'(\d{8})_(\d{4})', line)
                    if timestamp_match:
                        date_str, time_str = timestamp_match.groups()
                        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
                        dt += timedelta(minutes=60)  # Shift forward 60 minutes
                        completely_missing_dates[current_year].append(dt)
        
        # Find partially missing wavelengths
        missing_wavelengths_section = re.search(r'Days with missing wavelengths:.*?\n(.*?)(?:\n\n|$)', 
                                              section, re.DOTALL)
        if missing_wavelengths_section:
            for line in missing_wavelengths_section.group(1).strip().split('\n'):
                if 'SWPC Drawing Timestamp:' in line:
                    # Extract timestamp and wavelengths
                    timestamp_match = re.search(r'(\d{8})_(\d{4})', line)
                    wavelengths_match = re.search(r'Missing wavelengths: \[(.*?)\]', line)
                    
                    if timestamp_match and wavelengths_match:
                        date_str, time_str = timestamp_match.groups()
                        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
                        dt += timedelta(minutes=60)  # Shift forward 60 minutes
                        
                        wavelengths = {w.strip() for w in wavelengths_match.group(1).split(',')}
                        partially_missing_dates[current_year].append((dt, wavelengths))
    
    return completely_missing_dates, partially_missing_dates

async def download_missing_data(analysis_file: str, base_path: str):
    """
    Download missing wavelength data based on the analysis file.
    """
    completely_missing_dates, partially_missing_dates = await parse_analysis_file(analysis_file)
    
    # Process partially missing dates (only specific wavelengths)
    for year, date_wavelength_pairs in partially_missing_dates.items():
        print(f"\nProcessing partially missing dates for {year}")
        
        # Group by wavelength to minimize the number of download_multiple_wavelengths calls
        wavelength_to_dates = defaultdict(list)
        for dt, wavelengths in date_wavelength_pairs:
            for wavelength in wavelengths:
                wavelength_to_dates[wavelength].append(dt)
        
        # Download each wavelength group
        for wavelength, dates in wavelength_to_dates.items():
            print(f"Downloading wavelength {wavelength}Å for {len(dates)} dates")
            await download_multiple_wavelengths(
                base_path=base_path,
                wavelengths=[wavelength],
                dates=dates
            )

    # Process completely missing dates (need all wavelengths)
    for year, dates in completely_missing_dates.items():
        print(f"\nProcessing completely missing dates for {year}")
        if dates:
            wavelengths = ['94', '131', '171', '193', '211', '304', '335', '6173']
            await download_multiple_wavelengths(
                base_path=base_path,
                wavelengths=wavelengths,
                dates=dates
            )

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download missing wavelength data.')
    parser.add_argument('--analysis_file', type=str, required=True,
                       help='Path to the wavelength analysis output file')
    parser.add_argument('--base_path', type=str, required=True,
                       help='Base path for downloading files')
    
    args = parser.parse_args()
    
    await download_missing_data(args.analysis_file, args.base_path)

if __name__ == "__main__":
    asyncio.run(main())