import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import logging
import re
import time

from download import DataSetFetcher

def extract_datetime_from_filename(filename: str) -> datetime:
    """Extract datetime from filename of format boul_neutl_fd_YYYYMMDD_HHMM.jpg"""
    pattern = r'boul_neutl_fd_(\d{8})_(\d{4})\.jpg'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename {filename} doesn't match expected pattern")
    
    date_str, time_str = match.groups()
    full_datetime_str = f"{date_str}_{time_str}"
    return datetime.strptime(full_datetime_str, "%Y%m%d_%H%M")

def get_dates_from_directory(directory: Path) -> list[datetime]:
    """Scan directory for matching files and extract dates"""
    dates = []
    logging.info(f"Scanning directory: {directory}")
    
    for file_path in directory.glob("boul_neutl_fd_*.jpg"):
        try:
            date = extract_datetime_from_filename(file_path.name)
            dates.append(date)
            logging.debug(f"Extracted date {date} from {file_path.name}")
        except ValueError as e:
            logging.warning(f"Skipping file {file_path.name}: {str(e)}")
    
    dates.sort()
    logging.info(f"Found {len(dates)} valid dates in directory")
    return dates

async def collect_file_based_data(input_dir: Path, output_path: Path, n_workers=8, resolution=512):
    """Collect solar images based on dates extracted from files"""
    # Get dates from files
    dates = get_dates_from_directory(input_dir)
    if not dates:
        logging.error("No valid dates found in input directory")
        return
    
    logging.info(f"Preparing to collect data for {len(dates)} timestamps")
    logging.info(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Initialize fetcher
    fetcher = DataSetFetcher(
        ds_path=str(output_path),
        num_worker_threads=n_workers,
        resolution=resolution
    )
    
    # Fetch data for all dates
    await fetcher.fetchDates(dates)

async def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Collect solar images based on existing files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing boul_neutl_fd_*.jpg files')
    parser.add_argument('--output', type=str, default='solar_data',
                       help='Output directory for the collected data')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of dates to process in batch')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_path = Path(args.output) / "collection.log"
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logging.error(f"Input directory {input_path} does not exist")
        return
    
    logging.info(f"Input directory: {input_path}")
    logging.info(f"Output directory: {output_path}")
    logging.info(f"Using {args.workers} workers")
    
    try:
        # Run the collection
        await collect_file_based_data(
            input_path,
            output_path,
            n_workers=args.workers,
            resolution=512
        )
    finally:
        # Calculate and log the total running time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Convert to hours, minutes, seconds
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logging.info(f"Total running time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info(f"Total seconds elapsed: {total_time:.2f}")
        
        # Print to console as well
        print(f"\nTotal running time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Total seconds elapsed: {total_time:.2f}")
    
    logging.info("Data collection completed")

if __name__ == '__main__':
    asyncio.run(main())