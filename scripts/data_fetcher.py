#!/usr/bin/env python3
"""
Data Fetcher Module for Lotto Max ML Predictor
Manages historical draw data, including loading, updating, validation, and web scraping.

Supports two modes of operation:
1. Local mode: Load/save draws from the bundled assets/historical_draws.json file
2. Live mode: Fetch the latest draws from ca.lottonumbers.com via web scraping

Created by: Reza Azizi - May 2026
If you are interested in developing or enhancing an AI automation system—whether for workflows,
data processing, monitoring, or custom integrations—please feel free to reach out to
flowaiautomationsupport@gmail.com. I would be happy to discuss your requirements, propose a tailored solution,
and explore how automation can improve efficiency, accuracy, and scalability for your environment.

"""

import json
import os
import random
import re
from datetime import datetime, timedelta
from html.parser import HTMLParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, '..', 'assets')
DEFAULT_DRAWS_FILE = os.path.join(ASSETS_DIR, 'historical_draws.json')

NUM_POOL = 52
NUMS_PER_DRAW = 7
DATA_SOURCE_URL = "https://ca.lottonumbers.com/lotto-max/past-numbers"


# ─── Local Data Operations ─────────────────────────────────────────────────

def load_draws(filepath=None):
    """Load historical draw data from the bundled JSON file.
    
    The skill ships with pre-loaded historical data covering 6 months
    of real Lotto Max draws. No manual data entry is required.
    Users can optionally update the data using fetch_latest_draws().
    """
    if filepath is None:
        filepath = DEFAULT_DRAWS_FILE

    if not os.path.exists(filepath):
        print(f"Draw file not found: {filepath}")
        print("Generating sample data as fallback...")
        draws = generate_sample_draws(52)
        save_draws(draws, filepath)
        return draws

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        draws = data.get('draws', [])
        source = data.get('source', 'unknown')
        last_updated = data.get('last_updated', 'unknown')
        print(f"Loaded {len(draws)} draws (source: {source}, updated: {last_updated})")
        return draws
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return generate_sample_draws(52)


def save_draws(draws, filepath=None, source=None):
    """Save draw data to JSON file."""
    if filepath is None:
        filepath = DEFAULT_DRAWS_FILE

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = {
        'last_updated': datetime.now().strftime('%Y-%m-%d'),
        'source': source or DATA_SOURCE_URL,
        'total_draws': len(draws),
        'draws': draws
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(draws)} draws to {filepath}")


# ─── Web Scraping ──────────────────────────────────────────────────────────

class LottoMaxHTMLParser(HTMLParser):
    """Custom HTML parser to extract Lotto Max draw data from ca.lottonumbers.com."""
    
    def __init__(self):
        super().__init__()
        self.draws = []
        self.in_row = False
        self.in_date_cell = False
        self.in_balls_cell = False
        self.in_ball = False
        self.in_bonus_ball = False
        self.in_date_strong = False
        self.current_date_parts = []
        self.current_main_nums = []
        self.current_bonus = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'tr':
            self.in_row = True
            self.current_date_parts = []
            self.current_main_nums = []
            self.current_bonus = None
            
        elif tag == 'td':
            td_class = attrs_dict.get('class', '')
            if 'date-row' in td_class:
                self.in_date_cell = True
            elif 'balls-row' in td_class:
                self.in_balls_cell = True
                
        elif tag == 'strong' and self.in_date_cell:
            self.in_date_strong = True
            
        elif tag == 'li':
            li_class = attrs_dict.get('class', '')
            if 'bonus-ball' in li_class:
                self.in_bonus_ball = True
                self.in_ball = True
            elif 'ball' in li_class and 'bonus' not in li_class:
                self.in_ball = True
                
    def handle_endtag(self, tag):
        if tag == 'tr':
            # Process completed row
            if self.current_date_parts and len(self.current_main_nums) == 7:
                date_str = ' '.join(self.current_date_parts)
                try:
                    dt = datetime.strptime(date_str, "%B %d %Y")
                    self.draws.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'numbers': sorted(self.current_main_nums),
                        'bonus': self.current_bonus
                    })
                except ValueError:
                    pass
            self.in_row = False
            self.in_date_cell = False
            self.in_balls_cell = False
            
        elif tag == 'td':
            self.in_date_cell = False
            self.in_balls_cell = False
            
        elif tag == 'strong':
            self.in_date_strong = False
            
        elif tag == 'li':
            self.in_ball = False
            self.in_bonus_ball = False
            
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_date_cell and not self.in_date_strong:
            # This is the date part after <br> (e.g., "May 19 2026")
            self.current_date_parts.append(data)
            
        if self.in_ball and data.isdigit():
            num = int(data)
            if 1 <= num <= NUM_POOL:
                if self.in_bonus_ball:
                    self.current_bonus = num
                else:
                    self.current_main_nums.append(num)


def fetch_latest_draws(url=None, timeout=30):
    """Fetch the latest Lotto Max draws from ca.lottonumbers.com.
    
    This function scrapes the past 6 months of draw results from the
    official lottery numbers website. It requires the 'requests' library
    (pip install requests).
    
    Args:
        url: URL to fetch from (default: ca.lottonumbers.com Lotto Max page)
        timeout: Request timeout in seconds
        
    Returns:
        List of draw dicts, or None if fetch fails
    """
    if url is None:
        url = DATA_SOURCE_URL
        
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' library is required for web fetching.")
        print("Install it with: pip install requests")
        print("Alternatively, manually update assets/historical_draws.json")
        return None
    
    try:
        print(f"Fetching latest draws from {url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML
        parser = LottoMaxHTMLParser()
        parser.feed(response.text)
        draws = parser.draws
        
        if not draws:
            # Fallback: try regex-based parsing
            draws = _parse_with_regex(response.text)
        
        if draws:
            # Sort chronologically (oldest first)
            draws.sort(key=lambda x: x['date'])
            print(f"Successfully fetched {len(draws)} draws")
            print(f"  Date range: {draws[0]['date']} to {draws[-1]['date']}")
            return draws
        else:
            print("No draws found in the fetched page.")
            return None
            
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        print("Falling back to local data. Try again later or update manually.")
        return None


def _parse_with_regex(html):
    """Fallback regex-based parser for ca.lottonumbers.com HTML."""
    draws = []
    
    # Find table rows
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    
    for row in rows:
        # Extract date
        date_match = re.search(r'<strong>\w+</strong><br>(\w+\s+\d+\s+\d{4})', row)
        if not date_match:
            continue
        
        try:
            dt = datetime.strptime(date_match.group(1), "%B %d %Y")
            date_formatted = dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
        
        # Extract main numbers
        main_nums = [int(n) for n in re.findall(r'<li class="ball ball">(\d+)</li>', row)]
        
        # Extract bonus number
        bonus_match = re.findall(r'<li class="ball bonus-ball">(\d+)</li>', row)
        bonus = int(bonus_match[0]) if bonus_match else None
        
        if len(main_nums) == 7:
            draws.append({
                'date': date_formatted,
                'numbers': sorted(main_nums),
                'bonus': bonus
            })
    
    return draws


def update_data(filepath=None):
    """Fetch the latest draws from the web and update the local JSON file.
    
    This is the main entry point for keeping historical data current.
    The skill's bundled data covers 6 months, but users should run this
    periodically (e.g., weekly) to stay up to date.
    
    Usage:
        python scripts/data_fetcher.py --update
        python scripts/data_fetcher.py --update --url "https://ca.lottonumbers.com/lotto-max/past-numbers"
    """
    if filepath is None:
        filepath = DEFAULT_DRAWS_FILE
    
    # Load existing data
    existing_draws = []
    existing_dates = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            existing_draws = data.get('draws', [])
            existing_dates = {d['date'] for d in existing_draws}
        except:
            pass
    
    # Fetch new data
    new_draws = fetch_latest_draws()
    if new_draws is None:
        print("Could not fetch new data. Existing data unchanged.")
        return existing_draws
    
    # Merge: add any new draws not already in the dataset
    added = 0
    for draw in new_draws:
        if draw['date'] not in existing_dates:
            existing_draws.append(draw)
            existing_dates.add(draw['date'])
            added += 1
    
    # Sort all draws by date
    existing_draws.sort(key=lambda x: x['date'])
    
    # Save updated data
    save_draws(existing_draws, filepath)
    
    if added > 0:
        print(f"Added {added} new draws. Total: {len(existing_draws)}")
    else:
        print(f"No new draws to add. Total: {len(existing_draws)}")
    
    return existing_draws


# ─── Validation ────────────────────────────────────────────────────────────

def validate_draw(draw):
    """Validate a single draw entry."""
    if not isinstance(draw, dict):
        return False, "Draw must be a dictionary"

    if 'numbers' not in draw:
        return False, "Draw must contain 'numbers' key"

    numbers = draw['numbers']
    if not isinstance(numbers, list):
        return False, "Numbers must be a list"

    if len(numbers) != NUMS_PER_DRAW:
        return False, f"Must have exactly {NUMS_PER_DRAW} numbers, got {len(numbers)}"

    for n in numbers:
        if not isinstance(n, int) or n < 1 or n > NUM_POOL:
            return False, f"Number {n} out of range [1, {NUM_POOL}]"

    if len(set(numbers)) != NUMS_PER_DRAW:
        return False, "Duplicate numbers found"

    if 'bonus' in draw:
        bonus = draw['bonus']
        if not isinstance(bonus, int) or bonus < 1 or bonus > NUM_POOL:
            return False, f"Bonus number {bonus} out of range"
        if bonus in numbers:
            return False, "Bonus number cannot be in main numbers"

    return True, "Valid"


def add_draw(draws, date, numbers, bonus):
    """Add a new draw to the dataset."""
    draw = {
        'date': date,
        'numbers': sorted(numbers),
        'bonus': bonus
    }

    valid, message = validate_draw(draw)
    if not valid:
        print(f"Invalid draw: {message}")
        return draws

    draws.append(draw)
    return draws


# ─── Sample Data Generator (Fallback) ─────────────────────────────────────

def generate_sample_draws(count=52):
    """Generate realistic sample historical draws as fallback when no real data available.
    
    NOTE: This is a FALLBACK only. The skill ships with real historical data
    pre-loaded in assets/historical_draws.json. This generator is only used
    if the JSON file is missing or corrupted.
    """
    random.seed(42)
    draws = []
    base_date = datetime(2025, 12, 2)

    for i in range(count):
        draw_date = base_date + timedelta(days=i * 2.5)
        draw_date = draw_date.replace(hour=0, minute=0, second=0)

        numbers = sorted(random.sample(range(1, NUM_POOL + 1), NUMS_PER_DRAW))
        bonus_candidates = [n for n in range(1, NUM_POOL + 1) if n not in numbers]
        bonus = random.choice(bonus_candidates)

        draws.append({
            'date': draw_date.strftime('%Y-%m-%d'),
            'numbers': numbers,
            'bonus': bonus
        })

    random.seed()
    return draws


# ─── Statistics ────────────────────────────────────────────────────────────

def get_draw_statistics(draws):
    """Compute basic statistics about the draw dataset."""
    if not draws:
        return {'total': 0}

    dates = [d['date'] for d in draws]
    numbers = []
    for d in draws:
        numbers.extend(d['numbers'])

    return {
        'total': len(draws),
        'date_range': f"{dates[0]} to {dates[-1]}" if dates else "N/A",
        'total_numbers_drawn': len(numbers),
        'unique_numbers': len(set(numbers)),
        'avg_sum': round(sum(sum(d['numbers']) for d in draws) / len(draws), 2),
        'min_sum': min(sum(d['numbers']) for d in draws),
        'max_sum': max(sum(d['numbers']) for d in draws)
    }


# ─── CLI ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Lotto Max Data Fetcher')
    parser.add_argument('--update', action='store_true',
                       help='Fetch latest draws from the web and update local data')
    parser.add_argument('--url', default=DATA_SOURCE_URL,
                       help='URL to fetch draws from')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics about current data')
    parser.add_argument('--validate', action='store_true',
                       help='Validate all draws in the current dataset')
    args = parser.parse_args()
    
    print("Data Fetcher Module - Lotto Max ML Predictor")
    print("=" * 50)
    
    if args.update:
        update_data()
    elif args.stats:
        draws = load_draws()
        stats = get_draw_statistics(draws)
        print(f"\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    elif args.validate:
        draws = load_draws()
        invalid = 0
        for d in draws:
            valid, msg = validate_draw(d)
            if not valid:
                print(f"  Invalid draw on {d.get('date', 'unknown')}: {msg}")
                invalid += 1
        print(f"\nValidation: {len(draws) - invalid}/{len(draws)} draws valid")
    else:
        # Default: show current data info
        draws = load_draws()
        stats = get_draw_statistics(draws)
        print(f"\nCurrent dataset:")
        print(f"  Total draws: {stats['total']}")
        print(f"  Date range: {stats.get('date_range', 'N/A')}")
        print(f"  Average sum: {stats.get('avg_sum', 'N/A')}")
        print(f"\nTo update with latest draws: python scripts/data_fetcher.py --update")
