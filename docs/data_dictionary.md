# Data Dictionary

## Citizen Feedback Dataset

This document describes the structure and content of the citizen feedback dataset.

### Dataset Overview

- **Name**: Citizen Feedback on Public Services (Nigeria)
- **Type**: Synthetic dataset for demonstration and testing
- **Size**: ~50,000 records
- **Time Span**: 24 months (2024-01-01 to 2025-12-31)
- **Geographic Coverage**: All 37 states of Nigeria
- **Purpose**: Analyze citizen complaints, suggestions, and praise about public services

### Data Generation

This is a **synthetic dataset** generated to simulate realistic citizen feedback. While the data is not real, it is designed to reflect:
- Common feedback themes across Nigerian public services
- Realistic text patterns including Nigerian English expressions
- Distribution patterns similar to real-world feedback channels
- Typical data quality issues (duplicates, spam, missing values)

---

## Raw Data Fields

### `citizen_feedback.csv`

| Column | Type | Description | Example Values | Notes |
|--------|------|-------------|----------------|-------|
| `feedback_id` | String | Unique identifier for each feedback | FB000001, FB000002 | Format: FB + 6-digit number |
| `created_at` | Date | Date feedback was submitted | 2024-03-15 | ISO format (YYYY-MM-DD) |
| `state` | String | Nigerian state where feedback originated | Lagos, Kano, Rivers | One of 37 states |
| `lga` | String | Local Government Area | Ikeja, Kano Municipal | LGA names vary by state |
| `channel` | String | Feedback submission channel | SMS, Hotline, Web Form, In-person, Social Media | 5 possible channels |
| `facility_or_service` | String | Facility or service being discussed | Primary Health Center, Public School | Free text |
| `raw_text` | Text | Original feedback message | "The hospital is always closed..." | May contain typos, abbreviations |
| `rating` | Integer | Optional satisfaction rating | 1, 2, 3, 4, 5 | 1=worst, 5=best; empty if not provided |
| `response_time_days` | Integer | Days to first response | 5, 10, 15 | Only for resolved cases; empty otherwise |
| `resolved` | Boolean | Whether feedback was resolved | True, False | String representation |
| `assigned_dept` | String | Department assigned to handle feedback | Health, Education, Water, Sanitation, LocalGov, Other | Categorized automatically |

---

## Processed Data Fields

### `citizen_feedback_clean.parquet`

All fields from raw data, plus:

| Column | Type | Description | Range/Values | Notes |
|--------|------|-------------|--------------|-------|
| `cleaned_text` | Text | Cleaned and normalized text | N/A | PII masked, contractions expanded |
| `word_count` | Integer | Number of words in cleaned text | 2-500 | Messages with <2 words filtered out |
| `char_count` | Integer | Number of characters in cleaned text | 10-2000 | Approximate |
| `has_url` | Boolean | Whether text contains URL | True/False | Detected in original text |
| `has_number` | Boolean | Whether text contains numbers | True/False | Phone numbers, counts, etc. |
| `contains_expletive` | Boolean | Whether text contains mild expletives | True/False | Basic profanity detection |
| `lexical_diversity` | Float | Type-token ratio | 0.0-1.0 | Unique words / total words |
| `avg_word_length` | Float | Average word length in characters | 3.0-10.0 | char_count / word_count |

### `topic_assignments.csv`

All fields from cleaned data, plus:

| Column | Type | Description | Range/Values | Notes |
|--------|------|-------------|--------------|-------|
| `dominant_topic` | Integer | Most probable topic for document | 0-9 | Topic index (0 to n_topics-1) |
| `topic_0_prob` | Float | Probability of topic 0 | 0.0-1.0 | Document-topic probability |
| `topic_1_prob` | Float | Probability of topic 1 | 0.0-1.0 | " |
| ... | ... | ... | ... | One column per topic |
| `sentiment_score` | Float | Compound sentiment score | -1.0 to 1.0 | Negative to positive |
| `sentiment_label` | String | Sentiment category | positive, neutral, negative | Based on score thresholds |
| `sentiment_confidence` | Float | Confidence in sentiment | 0.0-1.0 | Absolute value of score |
| `polarity_score` | Float | Alternative polarity measure | -1.0 to 1.0 | Simple pos/neg word ratio |
| `polarity_label` | String | Alternative polarity label | positive, neutral, negative | Based on polarity |

---

## Data Quality Notes

### Known Issues (by design, to simulate real data)

1. **Duplicates**: ~3-5% of records are near-duplicates (copy-paste behavior)
2. **Spam**: ~2% of records contain nonsensical text
3. **Missing Values**:
   - `rating`: ~50-60% of SMS/Hotline entries lack ratings
   - `response_time_days`: Empty for unresolved cases (60% of records)
4. **Text Variations**:
   - Typos and abbreviations (e.g., "pls", "govt", "bcos")
   - Mixed case and punctuation
   - Nigerian English expressions (e.g., "abeg", "oga")

### PII Masking

Although this is synthetic data, PII-like patterns are masked to demonstrate best practices:
- Phone numbers → `[PHONE]`
- Email addresses → `[EMAIL]`
- National ID patterns → `[ID]`

---

## Channel Descriptions

| Channel | Description | Typical Volume | Rating Availability |
|---------|-------------|----------------|---------------------|
| SMS | Short text messages via mobile | High | Low (~40%) |
| Hotline | Phone calls transcribed | Medium | Low (~40%) |
| Web Form | Online submission form | Medium | High (~90%) |
| In-person | Walk-in submissions | Low | Medium (~60%) |
| Social Media | Twitter, Facebook mentions | Medium | Low (~30%) |

---

## Department Categories

| Department | Services Included | Example Facilities |
|------------|-------------------|-------------------|
| Health | Hospitals, clinics, PHCs | Primary Health Center, General Hospital |
| Education | Schools, education offices | Public School, Secondary School |
| Water | Water supply, boreholes | Water Supply, Borehole |
| Sanitation | Waste management, sanitation | Waste Management, Sanitation Dept |
| LocalGov | Local government services | Local Council, Ward Office, LGA Secretariat |
| Other | Police, fire, markets, roads | Police Station, Market, Road Maintenance |

---

## Topic Descriptions (Example)

Topics are discovered through LDA/NMF modeling. Example interpretations:

| Topic ID | Label | Top Terms | Description |
|----------|-------|-----------|-------------|
| 0 | Health Access | health, hospital, clinic, patient, medicine | Access issues at health facilities |
| 1 | Education | school, teacher, student, education | School-related feedback |
| 2 | Water Supply | water, supply, borehole, pipe | Water infrastructure issues |
| 3 | Staff Attitude | staff, rude, attitude, service | Complaints about staff behavior |
| 4 | Wait Times | wait, long, time, queue, hours | Long waiting times |
| 5 | Infrastructure | building, broken, repair, old | Infrastructure maintenance needs |
| 6 | Fees & Corruption | money, fee, pay, bribe, charge | Payment and corruption issues |
| 7 | Stockouts | medicine, drugs, available, stock | Medicine and supply shortages |
| 8 | Local Government | council, lga, ward, office | Local government services |
| 9 | Appreciation | thank, good, excellent, helpful | Positive feedback and praise |

---

## Usage Notes

1. **Date Handling**: All dates are in ISO format (YYYY-MM-DD). Convert to datetime for analysis.
2. **Boolean Fields**: Stored as strings ("True", "False"). Convert to boolean as needed.
3. **Missing Ratings**: Empty strings for missing ratings. Filter or convert to NaN.
4. **Text Analysis**: Use `cleaned_text` for NLP tasks, keep `raw_text` for reference.
5. **File Formats**:
   - CSV: Raw data (human-readable)
   - Parquet: Processed data (efficient storage, preserves types)

---

## Reproducibility

All data generation is seeded with `random_seed: 42` to ensure reproducibility. Running the generator with the same parameters will produce identical data.

---

## Contact & Support

For questions about the dataset structure or processing pipeline, refer to:
- README.md (project overview)
- modeling_notes.md (technical details)
- ethics_guidelines.md (responsible usage)
