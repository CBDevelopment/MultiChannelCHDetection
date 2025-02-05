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
from concurrent.futures import ThreadPoolExecutor

from chronnos.data.convert import prepMap

class DataSetFetcher:
    def __init__(self, ds_path, num_worker_threads=8, hmi_series='hmi.M_720s', resolution=None, max_retries=2):
        self.ds_path = ds_path
        self.resolution = resolution
        self.dirs = ['94', '131', '171', '193', '211', '304', '335', '6173']
        os.makedirs(ds_path, exist_ok=True)
        [os.makedirs(os.path.join(ds_path, dir), exist_ok=True) for dir in self.dirs]
        self.hmi_series = hmi_series
        self.max_retries = max_retries
        self.num_worker_threads = num_worker_threads
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f"{ds_path}/info_log.log"),
                logging.StreamHandler()
            ])
        
        self.drms_client = drms.Client()
        self.download_queue = asyncio.Queue()
        self.session = None
        self.semaphore = None
        self.map_executor = ThreadPoolExecutor(max_workers=4)  # For parallel map processing

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
        wavelength = int(header['WAVELNTH'])
        dir = os.path.join(self.ds_path, str(wavelength))
        obs_time = header['DATE__OBS']
        # Format timestamp as YYYY-MM-DDTHHMMSS
        formatted_time = pd.to_datetime(obs_time).strftime('%Y-%m-%dT%H%M%S')
        map_path = os.path.join(dir, f"{formatted_time}.fits")
        logging.info(f"Processing {wavelength}A data for {formatted_time}")

        if os.path.exists(map_path):
            logging.info(f'File exists: {map_path}, skipping.')
            return

        for attempt in range(self.max_retries):
            try:
                url = 'http://jsoc.stanford.edu' + segment
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
                # s_map.save(map_path)
                compressed_fits = CompImageHDU(
                    data=s_map.data, 
                    header=Header(s_map.meta),
                    compression_type='HCOMPRESS_1',
                    quantize_level=16.0)
                compressed_hdul = HDUList([PrimaryHDU(), compressed_fits])
                compressed_hdul.writeto(map_path)
                
                logging.info(f'Downloaded: {obs_time} / {header["WAVELNTH"]}')
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
        if not self.session:
            connector = aiohttp.TCPConnector(limit=self.num_worker_threads, force_close=True)
            self.session = aiohttp.ClientSession(connector=connector)
            self.semaphore = asyncio.Semaphore(self.num_worker_threads)
        
        workers = [asyncio.create_task(self.download_worker()) 
                  for _ in range(self.num_worker_threads)]
        
        try:
            for date in dates:
                if all([os.path.exists(os.path.join(self.ds_path, dir,
                                     date.isoformat('T', timespec='seconds').replace(':', '') + '.fits'))
                        for dir in self.dirs]):
                    logging.info(f"Skipping {date.isoformat()}, data exists.")
                    continue
                try:
                    await self.fetchData(date)
                except Exception as ex:
                    logging.error(f"Error fetching {date.isoformat()}: {traceback.format_exc()}")

            for _ in range(self.num_worker_threads):
                await self.download_queue.put(None)

            await self.download_queue.join()
            await asyncio.gather(*workers)
            
        finally:
            if self.session:
                await self.session.close()
                self.session = None
            self.map_executor.shutdown()

    async def fetchData(self, time):
        logging.info(f"\nFetching data for timestamp: {time.isoformat()}")
        time_param = f"{time.isoformat('_', timespec='seconds')}Z"
        
        # Query Magnetogram
        ds_hmi = f"{self.hmi_series}[{time_param}]{{magnetogram}}"
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_hmi, segment_hmi = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
        
        if len(header_hmi) != 1 or np.any(header_hmi.QUALITY != 0):
            logging.error(f'HMI data not valid for {time.isoformat()}. Using fallback.')
            await self.fetchDataFallback(time)
            return

        # Query EUV
        ds_euv = f'aia.lev1_euv_12s[{time_param}]{{image}}'
        keys_euv = self.drms_client.keys(ds_euv)
        header_euv, segment_euv = self.drms_client.query(ds_euv, key=','.join(keys_euv), seg='image')
        
        if len(header_euv) != 7 or np.any(header_euv.QUALITY != 0):
            logging.error(f'EUV data not valid for {time.isoformat()}. Using fallback.')
            await self.fetchDataFallback(time)
            return

        for (idx, h), s in zip(header_hmi.iterrows(), segment_hmi.magnetogram):
            await self.download_queue.put((h.to_dict(), s, time))
        for (idx, h), s in zip(header_euv.iterrows(), segment_euv.image):
            await self.download_queue.put((h.to_dict(), s, time))

    def query_euv(self, euv_ds, wavelength, time):
        """Query EUV data efficiently with optimized DataFrame handling."""
        keys_euv = self.drms_client.keys(euv_ds)
        header_tmp, segment_tmp = self.drms_client.query(euv_ds, key=','.join(keys_euv), seg='image')
        
        if len(header_tmp) == 0:
            raise ValueError(f'No data found for EUV wavelength {wavelength}')
            
        # Create date differences efficiently
        date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')
        date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - time).abs()
        
        # Create new DataFrames with all data at once
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
        
        # Apply quality filter
        quality_mask = header_df.QUALITY == 0
        header_df = header_df[quality_mask]
        segment_df = segment_df[quality_mask]
        
        if len(header_df) == 0:
            raise ValueError('No valid quality flag found')
        
        # Return first row without date_diff
        return (
            header_df.iloc[0].drop('date_diff'),
            segment_df.iloc[0].drop('date_diff')
        )

    async def fetchDataFallback(self, time):
        """Optimized fallback method for data fetching."""
        logging.info(f"\nUsing fallback method for {time.isoformat()}")
        t = time - timedelta(minutes=1)
        
        # Query Magnetogram
        ds_hmi = f'hmi.M_720s[{t.replace(tzinfo=None).isoformat("_", timespec="seconds")}Z/12h@720s]{{magnetogram}}'
        logging.info(f"Querying magnetogram with dataset: {ds_hmi}")
        
        keys_hmi = self.drms_client.keys(ds_hmi)
        header_tmp, segment_tmp = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
        
        if len(header_tmp) == 0:
            logging.error('No magnetogram data found in query')
            raise ValueError('No data found!')
        
        # Process magnetogram data efficiently
        date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')
        date_diff = np.abs(pd.to_datetime(date_str).dt.tz_localize(None) - time)
        
        # Create new DataFrames with all data at once
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
            logging.error("No valid HMI data found after quality filtering")
            raise ValueError("No valid HMI data found after filtering.")

        logging.info(f"Found valid magnetogram data with time difference: {header_df.iloc[0]['date_diff']}")
        header_hmi = header_df.iloc[0].drop('date_diff')
        segment_hmi = segment_df.iloc[0].drop('date_diff')

        # Query EUV with ThreadPoolExecutor
        header_euv, segment_euv = [], []
        t = time - timedelta(minutes=10)
        logging.info(f"Starting EUV queries for time: {t.isoformat()}")
        
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = []
            for wl in [94, 131, 171, 193, 211, 304, 335]:
                euv_ds = f'aia.lev1_euv_12s[{t.replace(tzinfo=None).isoformat("_", timespec="seconds")}Z/12h@12s][{wl}]{{image}}'
                logging.debug(f"Querying EUV wavelength {wl}Å with dataset: {euv_ds}")
                futures.append(executor.submit(self.query_euv, euv_ds, wl, t))

            for future in futures:
                h, s = future.result()
                header_euv.append(h)
                segment_euv.append(s)

        logging.info(f"Successfully retrieved all EUV data for {len(futures)} wavelengths")

        await self.download_queue.put((header_hmi.to_dict(), segment_hmi.magnetogram, time))
        logging.debug("Added magnetogram data to download queue")
        
        for h, s in zip(header_euv, segment_euv):
            await self.download_queue.put((h.to_dict(), s.image, time))
            logging.debug(f"Added EUV data for wavelength {h.get('WAVELNTH', 'unknown')}Å to download queue")
        
        logging.info(f"Completed fallback data fetch for {time.isoformat()}")

async def main():
    import time
    parser = argparse.ArgumentParser(description='Download AIA and HMI files for CHRONNNOS image segmentation.')
    parser.add_argument('--path', type=str, default=os.path.join(Path.home(), 'chronnos'))
    parser.add_argument('--dates', nargs='*', type=lambda s: parse(s))
    parser.add_argument('--hmi_series', type=str, default='hmi.M_720s')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=None)

    args = parser.parse_args()
    
    start_time = time.time()
    logging.info("Starting download process")
    
    fetcher = DataSetFetcher(
        ds_path=args.path,
        hmi_series=args.hmi_series,
        num_worker_threads=args.n_workers,
        resolution=args.resolution
    )
    await fetcher.fetchDates(args.dates)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logging.info(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logging.info("Download process complete")

if __name__ == '__main__':
    asyncio.run(main())