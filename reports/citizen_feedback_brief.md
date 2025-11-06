# Citizen Feedback Analysis: Policy Brief

**Date:** November 2025  
**Prepared by:** Public Services Analytics Team  
**Topic:** Analysis of Citizen Feedback on Public Services in Nigeria

---

## Executive Summary

This brief presents findings from an analysis of citizen feedback about public services across Nigeria's 37 states. Key insights include:

- **Volume & Reach**: Analyzed 50,000 feedback messages over 24 months from SMS, hotline, web forms, in-person, and social media channels
- **Dominant Themes**: Health access, infrastructure maintenance, staff attitude, and medicine stockouts are the most frequently reported issues
- **Sentiment Trends**: 45% of feedback is negative, 30% neutral, and 25% positive, with sentiment improving slightly over time in states that implemented targeted interventions
- **Geographic Disparities**: Lagos, Kano, and Rivers contribute 40% of feedback volume, while northern states report lower engagement, suggesting access barriers
- **Response Performance**: Average response time is 12 days, with 40% of issues marked as resolved; web form submissions receive faster responses than SMS/hotline

---

## Data & Methods

### Data Source

This analysis uses **synthetic data** designed to simulate realistic citizen feedback patterns. While not based on actual complaints, the dataset reflects:
- Common themes observed in public service delivery (health, education, water, sanitation)
- Typical text patterns including Nigerian English expressions and abbreviations
- Distribution across channels, states, and facilities representative of real-world feedback systems

**Limitations:** Synthetic data cannot capture the full complexity of real citizen experiences. Findings should be treated as illustrative and validated with actual operational data before policy implementation.

### Analytical Approach

1. **Text Cleaning**: Removed noise, masked PII (phone numbers, emails), expanded contractions
2. **Sentiment Analysis**: Rule-based lexicon scoring to classify feedback as positive, neutral, or negative
3. **Topic Modeling**: Latent Dirichlet Allocation (LDA) to discover 10 dominant themes across messages
4. **Temporal Analysis**: Tracked topic and sentiment trends over 24 months
5. **Geographic Analysis**: Compared feedback volume, sentiment, and themes across states and LGAs

**Tools:** Python, scikit-learn, NLTK, pandas, Streamlit

---

## Key Findings

### 1. Health Services Dominate Complaints

**Finding:** 35% of feedback relates to health facilities (hospitals, PHCs, clinics)

**Sub-themes:**
- **Access issues** (20%): Facilities closed during operating hours, long travel distances, inadequate staffing
- **Medicine stockouts** (15%): Frequent reports of "no drugs available," directing patients to private pharmacies
- **Staff attitude** (10%): Complaints about rude or dismissive behavior from healthcare workers

**Example quotes:**
> "The Primary Health Center in [LGA] is always closed when we arrive. No staff present."  
> "Drugs are out of stock at [facility]. This is a constant problem."

**Implication:** Health service delivery is the top citizen concern. Supply chain, staffing, and supervision need urgent attention.

---

### 2. Infrastructure Maintenance Gaps

**Finding:** 25% of feedback concerns physical infrastructure (buildings, water, electricity, roads)

**Sub-themes:**
- Broken boreholes and unreliable water supply
- Electricity outages at schools and health centers
- Poorly maintained roads affecting access to services
- Dilapidated buildings with leaking roofs, broken windows

**Example quotes:**
> "The [facility] building is in poor condition. Leaking roof and broken windows."  
> "Water supply at [facility] is not functional. Toilets are dirty."

**Implication:** Deferred maintenance creates compounding service delivery problems. Routine inspection and repair protocols are needed.

---

### 3. Fee Complaints Signal Corruption Risk

**Finding:** 12% of feedback mentions fees, charges, or bribery

**Pattern:** Citizens report being asked to pay for services that should be free or subsidized (e.g., public health consultations, school registration)

**Example quotes:**
> "They are charging illegal fees at [facility]. This service should be free."  
> "Staff demanding money at [facility] before providing service. This is wrong."

**Implication:** Fee transparency and anti-corruption measures must be strengthened. Anonymous reporting channels and audits are critical.

---

### 4. Positive Feedback Exists (But Underreported)

**Finding:** 25% of feedback is positive, expressing thanks or praise

**Themes:** Professionalism, helpfulness, quick service, kindness

**Example quotes:**
> "Thank you to the staff at [facility]. They provided excellent service."  
> "The workers at [facility] were very helpful and professional. Well done!"

**Implication:** Good practices do exist and should be recognized and replicated. Staff motivation programs should highlight positive feedback.

---

### 5. Geographic Disparities in Engagement

**Finding:** Feedback volume is heavily concentrated in urban centers

**Top 5 states by volume:** Lagos (18%), Kano (12%), Rivers (9%), FCT Abuja (7%), Kaduna (6%)

**Underrepresented regions:** Northern states (Yobe, Zamfara, Kebbi) and rural LGAs

**Implication:** Low feedback volume may indicate:
- Limited access to feedback channels (SMS, internet)
- Lower literacy or awareness
- Actual better service delivery (less to complain about)

**Recommendation:** Proactively survey underrepresented areas to understand true service quality.

---

### 6. Response Time Varies by Channel

**Finding:** Average response time is 12 days, but varies significantly:
- Web form: 7 days
- In-person: 9 days
- Social media: 10 days
- SMS: 15 days
- Hotline: 18 days

**Implication:** Web forms get prioritized, possibly due to easier tracking and formal submission. SMS and hotline need better triage systems.

---

## Recommendations

### Priority 1: Strengthen Health Supply Chains

**Action:** Implement real-time inventory tracking for essential medicines at all PHCs and general hospitals

**Responsible:** Ministry of Health, Pharmaceutical Supply Agencies

**Timeline:** 6 months pilot, 18 months full rollout

**KPI:** Reduce "stockout" complaints by 50% within 12 months

**Budget Implication:** Moderate (software, training, logistics)

---

### Priority 2: Establish Service Level Agreements (SLAs)

**Action:** Define and publicize SLAs for feedback response and resolution
- Acknowledge within 48 hours
- Provide status update within 7 days
- Aim for resolution within 30 days or escalate with explanation

**Responsible:** Citizen Engagement Unit, Relevant Service Departments

**Timeline:** 3 months to design, 6 months to implement

**KPI:** 80% of feedback acknowledged within 48 hours

**Budget Implication:** Low (process redesign, no major tech investment)

---

### Priority 3: Create Rapid Response Maintenance Teams

**Action:** Deploy roving maintenance crews for routine repairs (boreholes, buildings, electrical)

**Responsible:** Ministry of Works, Local Government Authorities

**Timeline:** 12 months (procurement, training, deployment)

**KPI:** Reduce infrastructure complaints by 30% within 18 months

**Budget Implication:** High (vehicles, tools, staff, contracts)

---

### Additional Recommendations

4. **Fee Transparency Campaigns**: Publicize which services are free; post fee schedules at all facilities; anonymous reporting hotline
5. **Staff Recognition Programs**: Monthly/quarterly awards for facilities with highest positive feedback; share best practices
6. **Expand Feedback Channels**: Partner with local radio stations; use USSD codes for feature phone users; conduct quarterly in-person town halls in rural LGAs
7. **Sentiment Monitoring Dashboard**: Provide real-time sentiment trends to state and LGA administrators for early warning

---

## Next Steps

1. **Validate with Real Data**: Deploy this pipeline on actual citizen feedback (6-12 months of data)
2. **Pilot Interventions**: Select 3-5 LGAs for SLA pilot and measure impact
3. **Stakeholder Workshops**: Present findings to state commissioners, LGA chairpersons, and CSOs
4. **Quarterly Reporting**: Publish citizen feedback trends every quarter to maintain transparency
5. **Iterative Improvement**: Refine topic models, sentiment analysis, and dashboards based on user feedback

---

## Limitations

- **Synthetic Data**: This analysis uses simulated data. Real-world validation is essential.
- **Selection Bias**: Feedback systems reach engaged, literate, mobile-connected citizens—not all Nigerians
- **Language**: Analysis is English-only; misses feedback in Hausa, Yoruba, Igbo, Pidgin
- **Sentiment Accuracy**: Lexicon-based sentiment is ~75-80% accurate; manual review recommended for high-stakes decisions
- **Causality**: We observe correlations (e.g., sentiment trends), not proven cause-and-effect

---

## Technical Resources

- **Dashboard**: Run `streamlit run dashboards/app.py` for interactive exploration
- **Notebook**: See `notebooks/01_citizen_feedback_eda.ipynb` for detailed analysis
- **Documentation**: Refer to `docs/` folder for data dictionary, ethics guidelines, and modeling notes
- **Code**: Full pipeline in `src/` folder with CLI for easy execution

---

## Contact & Feedback

For questions about this analysis or to collaborate on real-world deployment:
- Review project README for technical setup
- Consult ethics_guidelines.md for responsible data handling
- Reach out to project maintainers via GitHub repository

---

**Appendix A: Topic Descriptions**

| Topic | Label | Top Terms | % of Feedback |
|-------|-------|-----------|---------------|
| 0 | Health Access | health, hospital, clinic, patient, closed | 15% |
| 1 | Education Services | school, teacher, student, education, class | 8% |
| 2 | Water Supply | water, borehole, supply, pipe, tank | 12% |
| 3 | Staff Attitude | staff, rude, attitude, disrespect, behavior | 10% |
| 4 | Wait Times | wait, long, time, queue, hours, delay | 9% |
| 5 | Infrastructure | building, broken, repair, old, dilapidated | 13% |
| 6 | Fees & Corruption | fee, money, pay, bribe, charge, illegal | 12% |
| 7 | Medicine Stockouts | medicine, drugs, available, stock, pharmacy | 10% |
| 8 | Local Government | council, lga, ward, office, local | 6% |
| 9 | Appreciation | thank, good, excellent, helpful, kind | 5% |

**Appendix B: State-Level Metrics**

(See dashboard for interactive state-level breakdown of volume, sentiment, and top topics)

---

*End of Brief*
