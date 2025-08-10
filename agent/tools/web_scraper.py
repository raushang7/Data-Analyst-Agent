import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import re
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class WebScraper:
    """Tool for web scraping and data extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape data from a URL
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing scraped data
        """
        try:
            logger.info(f"Scraping URL: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if this is a Wikipedia page about movies
            if 'wikipedia.org' in url and 'highest-grossing' in url:
                return self._scrape_wikipedia_movies(soup, url)
            
            # General table extraction
            tables = soup.find_all('table')
            if tables:
                return self._extract_tables(tables, url)
            
            # Fallback to general text extraction
            return self._extract_general_content(soup, url)
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            raise
    
    def _scrape_wikipedia_movies(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape Wikipedia highest-grossing films data"""
        result = {
            'url': url,
            'title': soup.find('h1').get_text() if soup.find('h1') else '',
            'tables': [],
            'movies_data': []
        }
        
        # Find the main table with movie data
        tables = soup.find_all('table', class_='wikitable')
        
        for table in tables:
            try:
                # Convert table to DataFrame
                df = pd.read_html(str(table))[0]
                
                # Clean up the DataFrame
                if len(df.columns) > 5:  # Likely the main movies table
                    # Common column names in highest-grossing movies table
                    expected_cols = ['rank', 'peak', 'title', 'worldwide', 'year']
                    
                    # Try to map columns
                    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                    
                    # Extract relevant data
                    movies = []
                    for _, row in df.iterrows():
                        movie = {}
                        for col in df.columns:
                            if any(keyword in col.lower() for keyword in ['rank', 'peak', 'title', 'gross', 'year', 'worldwide']):
                                movie[col] = row[col]
                        
                        if movie:  # Only add if we found relevant data
                            movies.append(movie)
                    
                    result['movies_data'] = movies
                    result['tables'].append({
                        'data': df.to_dict('records'),
                        'columns': list(df.columns),
                        'shape': df.shape
                    })
                    
                    logger.info(f"Extracted {len(movies)} movies from Wikipedia")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to process table: {e}")
                continue
        
        return result
    
    def _extract_tables(self, tables: List, url: str) -> Dict[str, Any]:
        """Extract data from HTML tables"""
        result = {
            'url': url,
            'tables': []
        }
        
        for i, table in enumerate(tables):
            try:
                df = pd.read_html(str(table))[0]
                result['tables'].append({
                    'table_index': i,
                    'data': df.to_dict('records'),
                    'columns': list(df.columns),
                    'shape': df.shape
                })
            except Exception as e:
                logger.error(f"Failed to extract table {i}: {e}")
                continue
        
        return result
    
    def _extract_general_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract general content from a webpage"""
        result = {
            'url': url,
            'title': soup.find('title').get_text() if soup.find('title') else '',
            'headings': [],
            'paragraphs': [],
            'links': []
        }
        
        # Extract headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            result['headings'].append({
                'tag': heading.name,
                'text': heading.get_text().strip()
            })
        
        # Extract paragraphs
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if text:
                result['paragraphs'].append(text)
        
        # Extract links
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().strip()
            if text and href:
                result['links'].append({
                    'text': text,
                    'url': urljoin(url, href)
                })
        
        return result
    
    def extract_movie_data(self, scraped_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract and clean movie data from scraped Wikipedia data
        
        Args:
            scraped_data: Data scraped from Wikipedia
            
        Returns:
            Cleaned DataFrame with movie data
        """
        if 'movies_data' in scraped_data and scraped_data['movies_data']:
            df = pd.DataFrame(scraped_data['movies_data'])
        elif 'tables' in scraped_data and scraped_data['tables']:
            # Use the largest table
            largest_table = max(scraped_data['tables'], key=lambda x: x['shape'][0])
            df = pd.DataFrame(largest_table['data'])
        else:
            raise ValueError("No movie data found in scraped data")
        
        # Clean and standardize the data
        df = self._clean_movie_dataframe(df)
        return df
    
    def _clean_movie_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize movie DataFrame"""
        # Create a copy to avoid modifying original
        clean_df = df.copy()
        
        # Standardize column names
        column_mapping = {}
        for col in clean_df.columns:
            col_lower = str(col).lower()
            if 'rank' in col_lower:
                column_mapping[col] = 'Rank'
            elif 'peak' in col_lower:
                column_mapping[col] = 'Peak'
            elif 'title' in col_lower or 'film' in col_lower:
                column_mapping[col] = 'Title'
            elif 'worldwide' in col_lower or 'gross' in col_lower:
                column_mapping[col] = 'Worldwide_gross'
            elif 'year' in col_lower:
                column_mapping[col] = 'Year'
        
        clean_df = clean_df.rename(columns=column_mapping)
        
        # Clean numeric columns
        for col in ['Rank', 'Peak', 'Worldwide_gross']:
            if col in clean_df.columns:
                clean_df[col] = self._clean_numeric_column(clean_df[col])
        
        # Clean year column
        if 'Year' in clean_df.columns:
            clean_df['Year'] = self._extract_years(clean_df['Year'])
        
        # Clean title column
        if 'Title' in clean_df.columns:
            clean_df['Title'] = clean_df['Title'].astype(str).str.strip()
        
        return clean_df
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean numeric column by removing non-numeric characters"""
        def clean_value(val):
            if pd.isna(val):
                return None
            
            val_str = str(val)
            # Remove currency symbols, commas, and other non-numeric characters
            cleaned = re.sub(r'[^\d.]', '', val_str)
            
            try:
                return float(cleaned) if cleaned else None
            except:
                return None
        
        return series.apply(clean_value)
    
    def _extract_years(self, series: pd.Series) -> pd.Series:
        """Extract years from year column"""
        def extract_year(val):
            if pd.isna(val):
                return None
            
            val_str = str(val)
            # Look for 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', val_str)
            
            if year_match:
                return int(year_match.group())
            return None
        
        return series.apply(extract_year)