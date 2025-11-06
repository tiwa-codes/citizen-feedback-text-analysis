"""
Generate synthetic citizen feedback data for Nigeria.

This module creates realistic synthetic feedback data spanning multiple states,
LGAs, channels, and themes to simulate real citizen complaints and suggestions
about public services.
"""
import argparse
import uuid
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import csv

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.sampling_helpers import (
    NIGERIA_STATES, CHANNELS, FACILITIES_SERVICES, COMPLAINT_TEMPLATES,
    get_lgas_for_state, apply_text_variations, add_nigerian_flavor,
    generate_random_date, generate_spam_text, assign_rating,
    calculate_response_time, match_facility_to_department
)


def generate_feedback_text(theme: str, facility: str, lga: str, state: str) -> str:
    """
    Generate realistic feedback text based on theme.
    
    Args:
        theme: Feedback theme (e.g., 'access', 'staff_attitude')
        facility: Facility or service name
        lga: Local Government Area
        state: State name
        
    Returns:
        Generated feedback text
    """
    if theme not in COMPLAINT_TEMPLATES:
        theme = random.choice(list(COMPLAINT_TEMPLATES.keys()))
    
    template = random.choice(COMPLAINT_TEMPLATES[theme])
    text = template.format(facility=facility, lga=lga, state=state)
    
    # Apply variations and Nigerian patterns
    text = apply_text_variations(text, typo_prob=0.3)
    text = add_nigerian_flavor(text, prob=0.25)
    
    # Occasionally make text longer with additional context
    if random.random() < 0.3:
        additional_context = [
            "This has been happening for months.",
            "We have complained before but nothing changed.",
            "Other wards have same problem.",
            "Please help us urgently.",
            "This is affecting many people in the community.",
        ]
        text += " " + random.choice(additional_context)
    
    return text


def generate_single_feedback(
    feedback_id: int,
    start_date: datetime,
    months: int,
    duplicate_source: Dict = None
) -> Dict:
    """
    Generate a single feedback record.
    
    Args:
        feedback_id: Unique identifier
        start_date: Start date for random date generation
        months: Number of months to span
        duplicate_source: If provided, create near-duplicate of this record
        
    Returns:
        Dictionary containing feedback record
    """
    # Select state and LGA
    state = random.choice(NIGERIA_STATES)
    lgas = get_lgas_for_state(state)
    lga = random.choice(lgas)
    
    # Select channel
    channel = random.choices(
        CHANNELS,
        weights=[0.25, 0.20, 0.20, 0.15, 0.20],  # SMS slightly more common
        k=1
    )[0]
    
    # Select facility/service
    facility = random.choice(FACILITIES_SERVICES)
    
    # Determine if this is a duplicate (3-5% of records)
    is_duplicate = duplicate_source is not None
    
    if is_duplicate:
        # Create near-duplicate
        text = duplicate_source['raw_text']
        # Slight variation for near-duplicate
        if random.random() < 0.5:
            text = text.replace(".", "...").replace("Please", "Pls")
        theme = duplicate_source['theme']
        state = duplicate_source['state']
        lga = duplicate_source['lga']
        facility = duplicate_source['facility']
    else:
        # Generate new feedback
        # Weight towards complaints vs praise
        theme = random.choices(
            list(COMPLAINT_TEMPLATES.keys()),
            weights=[0.15, 0.15, 0.12, 0.12, 0.10, 0.12, 0.10, 0.14],  # praise weighted less
            k=1
        )[0]
        text = generate_feedback_text(theme, facility, lga, state)
    
    # Determine resolution status (40% resolved)
    resolved = random.random() < 0.4
    
    # Assign rating
    rating = assign_rating(channel, resolved)
    
    # Calculate response time
    response_time = calculate_response_time(resolved, channel)
    
    # Match department
    department = match_facility_to_department(facility)
    
    # Generate date
    created_at = generate_random_date(start_date, months)
    
    return {
        'feedback_id': f"FB{feedback_id:06d}",
        'created_at': created_at.strftime('%Y-%m-%d'),
        'state': state,
        'lga': lga,
        'channel': channel,
        'facility_or_service': facility,
        'raw_text': text,
        'rating': rating if rating > 0 else '',  # Empty string for missing ratings
        'response_time_days': response_time if resolved else '',
        'resolved': 'True' if resolved else 'False',
        'assigned_dept': department,
        'theme': theme  # Include for internal tracking (would not exist in real data)
    }


def generate_synthetic_feedback(
    n: int = 50000,
    months: int = 24,
    start_date: str = "2024-01-01",
    seed: int = 42
) -> List[Dict]:
    """
    Generate synthetic citizen feedback dataset.
    
    Args:
        n: Number of feedback records to generate
        months: Number of months to span
        start_date: Start date in ISO format
        seed: Random seed for reproducibility
        
    Returns:
        List of feedback dictionaries
    """
    random.seed(seed)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    
    feedback_records = []
    duplicate_sources = []
    
    print(f"Generating {n:,} synthetic feedback records...")
    print(f"Date range: {start_date} to {months} months")
    print(f"States: {len(NIGERIA_STATES)}")
    print(f"Random seed: {seed}")
    
    for i in range(n):
        # 3-5% duplicates
        if random.random() < 0.04 and len(duplicate_sources) > 0:
            source = random.choice(duplicate_sources)
            record = generate_single_feedback(i + 1, start_dt, months, duplicate_source=source)
        # 2% spam
        elif random.random() < 0.02:
            record = generate_single_feedback(i + 1, start_dt, months)
            record['raw_text'] = generate_spam_text()
            record['theme'] = 'spam'
        else:
            record = generate_single_feedback(i + 1, start_dt, months)
            # Store for potential duplication
            if len(duplicate_sources) < 1000:
                duplicate_sources.append(record)
        
        feedback_records.append(record)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1:,} records...")
    
    print(f"✓ Successfully generated {len(feedback_records):,} records")
    
    return feedback_records


def save_to_csv(records: List[Dict], output_path: Path):
    """
    Save feedback records to CSV file.
    
    Args:
        records: List of feedback dictionaries
        output_path: Path to output CSV file
    """
    # Define field order (exclude 'theme' from output as it's internal)
    fieldnames = [
        'feedback_id', 'created_at', 'state', 'lga', 'channel',
        'facility_or_service', 'raw_text', 'rating', 'response_time_days',
        'resolved', 'assigned_dept'
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in records:
            # Filter out 'theme' field
            row = {k: v for k, v in record.items() if k in fieldnames}
            writer.writerow(row)
    
    print(f"✓ Saved to {output_path}")


def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic citizen feedback data for Nigeria"
    )
    parser.add_argument(
        '--n',
        type=int,
        default=50000,
        help='Number of feedback records to generate (default: 50000)'
    )
    parser.add_argument(
        '--months',
        type=int,
        default=24,
        help='Number of months to span (default: 24)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Start date in YYYY-MM-DD format (default: 2024-01-01)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/citizen_feedback.csv',
        help='Output CSV file path (default: data/raw/citizen_feedback.csv)'
    )
    
    args = parser.parse_args()
    
    # Resolve output path
    if Path(args.output).is_absolute():
        output_path = Path(args.output)
    else:
        # Relative to project root
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / args.output
    
    # Generate data
    records = generate_synthetic_feedback(
        n=args.n,
        months=args.months,
        start_date=args.start_date,
        seed=args.seed
    )
    
    # Save to CSV
    save_to_csv(records, output_path)
    
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"{'='*60}")
    print(f"Records: {len(records):,}")
    print(f"Output: {output_path}")
    print(f"States covered: {len(NIGERIA_STATES)}")
    print(f"Channels: {', '.join(CHANNELS)}")
    print(f"\nNext steps:")
    print(f"  1. Run cleaning: python -m src.text.cleaning")
    print(f"  2. Explore data: jupyter lab notebooks/01_citizen_feedback_eda.ipynb")


if __name__ == '__main__':
    main()
