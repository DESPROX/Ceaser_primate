import json
import os
from pathlib import Path
import logging
from typing import Set, List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from dataset import ArxivDownloader, AdaptiveRateLimiter
from datetime import datetime
import sys
import re


@dataclass
class DownloadStats:
    """Statistics for paper downloads."""
    total_papers: int
    downloaded_pdfs: int
    removed_papers: int
    missing_papers: int

    def completion_rate(self) -> float:
        """Calculate completion rate as percentage."""
        if self.total_papers == 0:
            return 0.0
        return (self.downloaded_pdfs / self.total_papers) * 100

    def validate_consistency(self) -> bool:
        """Check if statistics are consistent."""
        return (self.downloaded_pdfs + self.removed_papers + self.missing_papers) == self.total_papers


class PaperDownloadManager:
    """Manages the downloading and tracking of research papers."""
    
    def __init__(self,
                 metadata_file: str = 'lhcb_papers.json',
                 pdf_dir: str = 'lhcb_pdfs',
                 log_file: str = 'arxiv_download.log',
                 removed_papers_file: str = 'removed_papers.json'):
        
        self.metadata_file = Path(metadata_file)
        self.pdf_dir = Path(pdf_dir)
        self.log_file = Path(log_file)
        self.removed_papers_file = Path(removed_papers_file)

        self._setup_logging()
        
        self.pdf_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file), 
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_metadata(self) -> List[Dict]:
        """Load the papers metadata with error handling."""
        try: 
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"Metadata file {self.metadata_file} not found")
            
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Metadata should be a list of paper objects")
            
            for i, paper in enumerate(data):
                if not isinstance(paper, dict) or 'id' not in paper:
                    raise ValueError(f"Paper at index {i} missing required 'id' field")
            
            self.logger.info(f"Loaded {len(data)} papers from metadata")
            return data
        
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error loading metadata: {e}")
            return []
        
    def get_downloaded_pdfs(self) -> Set[str]:
        """Get set of downloaded PDFs with improved ID parsing."""
        if not self.pdf_dir.exists():
            self.logger.warning(f"PDF directory {self.pdf_dir} does not exist")
            return set()
        
        pdfs = set()
        pdf_files = list(self.pdf_dir.glob('*.pdf'))

        for pdf_path in pdf_files:
            try:
                paper_id = pdf_path.stem.replace('_', '/')
                pdfs.add(paper_id)
            except Exception as e:
                self.logger.warning(f"Could not parse PDF filename {pdf_path.name}: {e}")

        self.logger.info(f"Found {len(pdfs)} downloaded PDFs")
        return pdfs
    
    def parse_log_for_404s(self) -> Set[str]:
        """Parse log file for papers that returned 404 with improved regex."""
        removed_papers = set()

        if not self.log_file.exists():
            self.logger.warning(f"Log file {self.log_file} not found")
            return removed_papers
        
        try: 
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if '404' in line and ('removed' in line.lower() or 'retracted' in line.lower() or 'not found' in line.lower()):
                        patterns = [
                            r'Paper\s+([^\s]+)\s+not found',
                            r'404.*?([a-zA-Z-]+/\d+v?\d*)',
                            r'404.*?(\d{4}\.\d{4}v?\d*)',
                        ]

                        for pattern in patterns:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match: 
                                paper_id = match.group(1)
                                removed_papers.add(paper_id)
                                self.logger.debug(f"Found removed paper from log: {paper_id}")
                                break

        except Exception as e:
            self.logger.error(f"Error parsing log file: {e}")

        self.logger.info(f"Found {len(removed_papers)} removed papers from log")
        return removed_papers
    
    def load_removed_papers_json(self) -> Set[str]:
        """Load removed papers from JSON file."""
        if not self.removed_papers_file.exists():
            return set()
        
        try:
            with open(self.removed_papers_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                removed = set(data)
            elif isinstance(data, dict) and 'removed_papers' in data:
                removed = set(data['removed_papers'])
            else:
                self.logger.warning("Unexpected format in removed_papers.json")
                return set()
                
            self.logger.info(f"Loaded {len(removed)} removed papers from JSON")
            return removed 
            
        except (json.JSONDecodeError, KeyError) as e: 
            self.logger.error(f"Error loading removed papers JSON: {e}")
            return set()
        
    def save_removed_papers(self, removed_papers: Set[str]) -> None:
        """Save removed papers to JSON file with timestamp."""
        data = {
            'removed_papers': list(removed_papers),
            'last_updated': datetime.now().isoformat(), 
            'count': len(removed_papers)
        }

        try:
            with open(self.removed_papers_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(removed_papers)} removed papers to {self.removed_papers_file}")
        except Exception as e:
            self.logger.error(f"Error saving removed papers: {e}")
    
    def calculate_stats(self, papers: List[Dict]) -> Tuple[DownloadStats, Set[str]]:
        """Calculate comprehensive download statistics."""
        paper_ids = {paper['id'] for paper in papers}
        downloaded_pdfs = self.get_downloaded_pdfs()
        removed_papers_log = self.parse_log_for_404s()
        removed_papers_json = self.load_removed_papers_json()
        removed_papers = removed_papers_log | removed_papers_json

        if removed_papers_log - removed_papers_json:
            self.save_removed_papers(removed_papers)

        missing_papers = paper_ids - downloaded_pdfs - removed_papers
        
        stats = DownloadStats(
            total_papers=len(papers), 
            downloaded_pdfs=len(downloaded_pdfs),
            removed_papers=len(removed_papers),
            missing_papers=len(missing_papers)
        )

        return stats, missing_papers
    
    def print_statistics(self, stats: DownloadStats) -> None:
        """Print formatted statistics."""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total papers in metadata: {stats.total_papers:,}")
        print(f"Downloaded PDFs:         {stats.downloaded_pdfs:,}")
        print(f"Removed/retracted:       {stats.removed_papers:,}")
        print(f"Missing papers:          {stats.missing_papers:,}")
        print(f"Completion rate:         {stats.completion_rate():.1f}%")
        
        if not stats.validate_consistency():
            print(f"\nWARNING: Statistics inconsistency detected!")
            expected = stats.downloaded_pdfs + stats.removed_papers + stats.missing_papers
            print(f"Sum of components ({expected:,}) != Total papers ({stats.total_papers:,})")
        else:
            print(f"\nStatistics are consistent")

    def download_missing_papers(self,
                                missing_papers: Set[str],
                                batch_size: int = 10, 
                                initial_delay: float = 5.0,
                                max_delay: float = 300.0) -> Tuple[List[str], List[str]]:
        """Download missing papers with enhanced configuration."""
        if not missing_papers:
            self.logger.info("No missing papers to download")
            return [], []
         
        papers_to_download = [{'id': paper_id} for paper_id in missing_papers]
        rate_limiter = AdaptiveRateLimiter(
            initial_delay=initial_delay, 
            max_delay=max_delay
        )

        downloader = ArxivDownloader(rate_limiter=rate_limiter)

        self.logger.info(f"Starting download of {len(papers_to_download)} missing papers")
        print(f"\nDownloading {len(papers_to_download)} missing papers...")
        print(f"Batch size: {batch_size}")
        print(f"Output directory: {self.pdf_dir}")

        try:
            successful, failed = downloader.process_batch(
                papers_to_download,
                self.pdf_dir,
                batch_size=batch_size
            )
            
            return successful, failed
            
        except Exception as e:
            self.logger.error(f"Error during batch download: {e}")
            return [], list(missing_papers)

    def print_download_results(self, successful: List[str], failed: List[str]) -> None:
        """Print formatted download results."""
        print("\n" + "="*50)
        print("DOWNLOAD RESULTS")
        print("="*50)
        print(f"Successfully downloaded: {len(successful):,}")
        print(f"Failed downloads:        {len(failed):,}")
        
        if successful:
            print(f"\nSample successful downloads:")
            for paper_id in sorted(successful)[:5]: 
                print(f"   - {paper_id}")
            if len(successful) > 5:
                print(f"   ... and {len(successful) - 5} more")
        
        if failed:
            print(f"\nFailed papers:")
            for paper_id in sorted(failed):
                print(f"   - {paper_id}")

    def interactive_download_prompt(self, missing_papers: Set[str]) -> Union[bool, str]:
        """Interactive prompt for downloading missing papers."""
        if not missing_papers:
            return False
            
        print(f"\nFound {len(missing_papers)} missing papers")
        sample_size = min(10, len(missing_papers))
        sample_papers = sorted(list(missing_papers))[:sample_size]
        print(f"\nSample missing papers:")
        for paper_id in sample_papers:
            print(f"   - {paper_id}")
        if len(missing_papers) > sample_size:
            print(f"   ... and {len(missing_papers) - sample_size} more")
        
        while True:
            response = input(f"\nDownload {len(missing_papers)} missing papers? (y/n/s for sample): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            elif response in ['s', 'sample']:
                return 'sample'
            else:
                print("Please enter 'y' for yes, 'n' for no, or 's' for sample")

    def run(self) -> None:
        """Main execution method."""
        print("Starting Paper Download Manager")
        papers = self.load_metadata()
        if not papers:
            print("No papers loaded. Exiting.")
            return
        stats, missing_papers = self.calculate_stats(papers)
        self.print_statistics(stats)
        if missing_papers:
            download_choice = self.interactive_download_prompt(missing_papers)
            
            if download_choice == 'sample':
                sample_papers = set(list(missing_papers)[:5])
                print(f"\nDownloading sample of {len(sample_papers)} papers...")
                successful, failed = self.download_missing_papers(sample_papers)
            elif download_choice:
                successful, failed = self.download_missing_papers(missing_papers)
            else:
                print("Skipping download. Run again when ready to download.")
                return
        
            self.print_download_results(successful, failed)
            
            if failed:
                print(f"\nUpdating removed papers list with {len(failed)} failed downloads...")
                removed_papers = self.parse_log_for_404s() | self.load_removed_papers_json()
                removed_papers.update(failed)
                self.save_removed_papers(removed_papers)
        else:
            print("\nAll papers are already downloaded!")


def main():
    """Main function with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and manage research papers')
    parser.add_argument('--metadata', default='lhcb_papers.json', 
                       help='Path to metadata JSON file')
    parser.add_argument('--pdf-dir', default='lhcb_pdfs', 
                       help='Directory for PDF downloads')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for downloads')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive prompts')
    
    args = parser.parse_args()
    
    try:
        manager = PaperDownloadManager(
            metadata_file=args.metadata,
            pdf_dir=args.pdf_dir
        )
        
        if args.no_interactive:
            papers = manager.load_metadata()
            if papers:
                stats, missing_papers = manager.calculate_stats(papers)
                manager.print_statistics(stats)
        else:
            manager.run()
            
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()