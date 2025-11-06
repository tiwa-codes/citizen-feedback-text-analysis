"""
Helper functions for sampling and generating synthetic data.
"""
import random
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


# Nigerian states (exact names as specified)
NIGERIA_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", 
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", 
    "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", 
    "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", 
    "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara", "FCT Abuja"
]

# Sample LGAs for each state (representative, not exhaustive)
STATE_LGAS: Dict[str, List[str]] = {
    "Lagos": ["Ikeja", "Surulere", "Alimosho", "Oshodi-Isolo", "Lagos Island", "Epe"],
    "Kano": ["Kano Municipal", "Gwale", "Dala", "Nassarawa", "Fagge", "Tarauni"],
    "Rivers": ["Port Harcourt", "Obio-Akpor", "Eleme", "Ikwerre", "Oyigbo"],
    "FCT Abuja": ["Abuja Municipal", "Gwagwalada", "Kuje", "Bwari", "Kwali", "Abaji"],
    "Kaduna": ["Kaduna North", "Kaduna South", "Zaria", "Sabon Gari", "Chikun"],
    # Add generic LGAs for other states
}

# Generic LGAs for states not explicitly defined
GENERIC_LGAS = ["Central", "North", "South", "East", "West", "Municipal"]

# Channels
CHANNELS = ["SMS", "Hotline", "Web Form", "In-person", "Social Media"]

# Facilities and services
FACILITIES_SERVICES = [
    "Primary Health Center", "General Hospital", "Maternity Ward", "Clinic",
    "Public School", "Secondary School", "Education Office", "Adult Education Center",
    "Water Supply", "Borehole", "Water Treatment Plant",
    "Local Council", "Ward Office", "LGA Secretariat", "Town Hall",
    "Sanitation Department", "Waste Management", "Market", "Motor Park",
    "Police Station", "Fire Service", "Road Maintenance"
]

# Departments
DEPARTMENTS = ["Health", "Education", "Water", "Sanitation", "LocalGov", "Other"]

# Complaint themes and templates
COMPLAINT_TEMPLATES = {
    "access": [
        "The {facility} in {lga} is always closed when we arrive. No staff present.",
        "We traveled far to reach the {facility} but it was locked. Please improve access.",
        "The {facility} opens very late. Many people wait hours before services start.",
        "No proper access to {facility}. The road is bad and the gate is locked most times.",
    ],
    "staff_attitude": [
        "Staff at {facility} are very rude. They shout at patients and show no respect.",
        "The workers at {facility} treat us badly. Very poor customer service.",
        "Staff members are not helpful at {facility}. They ignore our questions.",
        "Rude behavior from staff at the {facility}. We deserve better treatment.",
    ],
    "wait_times": [
        "Waiting time at {facility} is too long. Spent 5 hours just to register.",
        "Very long queues at {facility}. No proper queue management system.",
        "We wait for many hours at {facility}. Process is too slow.",
        "The wait time at {facility} is unbearable. Need faster service.",
    ],
    "stockouts": [
        "No medicine available at {facility}. Always telling us to buy from pharmacy.",
        "Drugs are out of stock at {facility}. This is a constant problem.",
        "The {facility} has no supplies. No syringes, no bandages, nothing.",
        "Medicine shortage at {facility}. We need government to supply drugs.",
    ],
    "fees": [
        "They are charging illegal fees at {facility}. This service should be free.",
        "Too many charges at {facility}. Corruption everywhere.",
        "Staff demanding money at {facility} before providing service. This is wrong.",
        "Hidden fees at {facility}. They ask for money for everything.",
    ],
    "infrastructure": [
        "The {facility} building is in poor condition. Leaking roof and broken windows.",
        "No electricity at {facility}. Generator not working for weeks.",
        "Water supply at {facility} is not functional. Toilets are dirty.",
        "The {facility} needs renovation. Equipment is old and broken.",
    ],
    "praise": [
        "Thank you to the staff at {facility}. They provided excellent service.",
        "The workers at {facility} were very helpful and professional. Well done!",
        "Good experience at {facility}. The nurse was kind and thorough.",
        "I appreciate the quick service at {facility}. Keep up the good work.",
    ],
    "info_request": [
        "Please I need information about registration at {facility}.",
        "How can I access services at {facility}? Need guidance.",
        "What documents are needed for {facility}? Please advise.",
        "I want to know the operating hours of {facility}.",
    ],
}

# Common Nigerian English patterns and abbreviations
NIGERIAN_PATTERNS = [
    "abeg", "oga", "madam", "sir", "pls", "plz", "govt", "LGA", "PHC",
    "bcos", "cos", "dnt", "dont", "wont", "cant", "shld", "shd", "pple"
]

# Common typos and variations
TYPO_PATTERNS = {
    "the": ["d", "teh", "the"],
    "please": ["pls", "plz", "please", "plss"],
    "because": ["bcos", "cos", "bcuz", "because"],
    "people": ["pple", "peple", "people"],
    "should": ["shld", "shd", "should"],
    "government": ["govt", "gov't", "government"],
}


def get_lgas_for_state(state: str) -> List[str]:
    """Get list of LGAs for a given state."""
    if state in STATE_LGAS:
        return STATE_LGAS[state]
    return [f"{state} {lga}" for lga in GENERIC_LGAS]


def apply_text_variations(text: str, typo_prob: float = 0.3) -> str:
    """
    Apply Nigerian English patterns and occasional typos to make text more realistic.
    
    Args:
        text: Original text
        typo_prob: Probability of applying typo/variation to a word
        
    Returns:
        Modified text with variations
    """
    words = text.split()
    modified_words = []
    
    for word in words:
        word_lower = word.lower().strip('.,!?')
        
        # Apply typo patterns
        if word_lower in TYPO_PATTERNS and random.random() < typo_prob:
            word = random.choice(TYPO_PATTERNS[word_lower])
        
        modified_words.append(word)
    
    return ' '.join(modified_words)


def add_nigerian_flavor(text: str, prob: float = 0.2) -> str:
    """
    Add Nigerian English patterns to text.
    
    Args:
        text: Original text
        prob: Probability of adding Nigerian pattern
        
    Returns:
        Modified text
    """
    if random.random() < prob:
        # Add at beginning
        if random.random() < 0.5:
            pattern = random.choice(["Abeg", "Please", "Oga", "Sir", "Madam"])
            text = f"{pattern}, {text.lower()}"
    
    return text


def generate_random_date(start_date: datetime, months: int) -> datetime:
    """
    Generate a random date within specified range.
    
    Args:
        start_date: Starting date
        months: Number of months to span
        
    Returns:
        Random datetime
    """
    days_range = months * 30
    random_days = random.randint(0, days_range)
    return start_date + timedelta(days=random_days)


def generate_spam_text() -> str:
    """Generate nonsensical spam text."""
    spam_patterns = [
        "asdfghjkl qwerty zxcvbn",
        "123456789 test test test",
        "xxxxxxxxxx yyyyyyy zzzzz",
        "click here for free money!!!",
        "win lottery now call +234",
        "aaaaaa bbbbbb cccccc ddddd",
    ]
    return random.choice(spam_patterns)


def assign_rating(channel: str, resolved: bool) -> int:
    """
    Assign a rating based on channel and resolution status.
    Many SMS entries lack ratings.
    
    Args:
        channel: Feedback channel
        resolved: Whether feedback was resolved
        
    Returns:
        Rating 1-5 or 0 if not available
    """
    # SMS and Hotline often lack ratings
    if channel in ["SMS", "Hotline"] and random.random() < 0.6:
        return 0  # No rating
    
    # Resolved issues tend to have better ratings
    if resolved:
        return random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
    else:
        return random.choices([1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.1])[0]


def calculate_response_time(resolved: bool, channel: str) -> int:
    """
    Calculate response time in days.
    
    Args:
        resolved: Whether feedback was resolved
        channel: Feedback channel
        
    Returns:
        Response time in days (0 if not resolved)
    """
    if not resolved:
        return 0
    
    # Web forms and social media tend to get faster responses
    if channel in ["Web Form", "Social Media"]:
        return random.randint(1, 15)
    elif channel == "In-person":
        return random.randint(1, 10)
    else:  # SMS, Hotline
        return random.randint(5, 30)


def match_facility_to_department(facility: str) -> str:
    """Match facility/service to appropriate department."""
    facility_lower = facility.lower()
    
    if any(word in facility_lower for word in ["health", "hospital", "clinic", "phc", "maternity"]):
        return "Health"
    elif any(word in facility_lower for word in ["school", "education"]):
        return "Education"
    elif any(word in facility_lower for word in ["water", "borehole"]):
        return "Water"
    elif any(word in facility_lower for word in ["sanitation", "waste"]):
        return "Sanitation"
    elif any(word in facility_lower for word in ["council", "lga", "ward", "town hall"]):
        return "LocalGov"
    else:
        return "Other"
