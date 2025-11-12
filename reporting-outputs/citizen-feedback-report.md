# Citizen Feedback Analysis: Donor Report

**Reporting Period:** January 2024 – December 2025  
**Total Feedback Entries:** 100  
**Date Prepared:** November 2025  
**Project:** Nigeria Citizen Feedback Text Analysis

---

## Executive Summary

This report presents findings from a systematic analysis of citizen feedback collected through multiple channels across Nigeria. The analysis employs natural language processing and topic modeling to identify key themes, sentiment patterns, and actionable insights for service delivery improvement.

**Key Highlights:**
- 100 feedback entries analyzed from 34 states
- 53% negative sentiment indicates significant service delivery challenges
- Infrastructure and equipment issues dominate citizen concerns (25%)
- Staff availability and facility closure affect service access
- Multi-channel feedback system captures diverse citizen voices

---

## 1. What We Found

### Dataset Overview

The analysis covers feedback received through five channels (SMS, Hotline, Web Form, In-person, and Social Media) spanning 23 months from January 2024 to December 2025. The dataset represents citizens across 34 Nigerian states reporting on health, education, water, sanitation, and local government services.

**Geographic Coverage:**
- States represented: 34 out of 37
- Urban and rural Local Government Areas (LGAs)
- Multi-channel submission enabling diverse citizen participation

### Sentiment Analysis

Sentiment analysis reveals concerning patterns in citizen satisfaction:

| Sentiment | Percentage | Count |
|-----------|------------|-------|
| Negative  | 53%        | 53    |
| Neutral   | 37%        | 37    |
| Positive  | 10%        | 10    |

**Key Insight:** The high negative sentiment (53%) signals systemic service delivery challenges requiring urgent attention. Only 10% of feedback expresses satisfaction, indicating limited success stories to build upon.

### Top 5 Topics Identified

Using Latent Dirichlet Allocation (LDA) topic modeling, we identified five dominant themes in citizen feedback:

#### **Topic 1: Infrastructure & Equipment Issues (25%)**
Citizens report deteriorating physical infrastructure including broken equipment, leaking roofs, and non-functional facilities.

**Example Quote:**  
> "Please, the clinic building is in poor condition. leaking roof and broken windows."

#### **Topic 2: Wait Times & Customer Service (24%)**
Long wait times and poor service quality are persistent complaints across service points.

**Example Quote:**  
> "The Ward Office opens very late. Many people wait hours before services start."

#### **Topic 3: Access & Service Availability (21%)**
Barriers to accessing services including locked gates, bad roads, and unclear procedures.

**Example Quote:**  
> "How can I access services at Adult Education Center? Need guidance. This has been happening for months."

#### **Topic 4: Staff Presence & Facility Closure (18%)**
Facilities closed during operating hours with no staff present is a recurring frustration.

**Example Quote:**  
> "Abeg, the adult education center in taraba east is always closed when we arrive. no staff present."

#### **Topic 5: Fees & Appreciation (12%)**
Mix of complaints about illegal fees and expressions of gratitude where service is good.

**Example Quote:**  
> "Oga, staff demanding money at town hall before providing service. this is wrong."

---

## 2. What It Means

### Service Delivery Crisis

The predominance of negative feedback (53%) combined with infrastructure complaints (25%) points to a service delivery system under significant strain. Citizens are experiencing:

1. **Physical deterioration** of facilities affecting service quality
2. **Human resource gaps** with staff absent or insufficient
3. **Process inefficiencies** causing long waits and poor experiences
4. **Accountability issues** including informal fee demands

### Geographic and Access Disparities

While 34 states are represented, feedback volume likely correlates with literacy, mobile access, and urban concentration. The analysis may under-represent:
- Rural and remote communities
- Populations with limited literacy or technology access
- Northern states with lower feedback volumes

### Positive Signals Amid Challenges

Despite challenges, 10% positive feedback indicates pockets of good practice. These examples should be:
- Documented and studied
- Replicated across similar service points
- Used to motivate and recognize performing staff

---

## 3. Recommended Next Steps

### Priority 1: Infrastructure Rapid Response Program

**Action:** Establish maintenance teams for quick repairs to critical infrastructure (buildings, water points, equipment)

**Rationale:** 25% of complaints relate to physical infrastructure; visible improvements build citizen trust

**Timeline:** 6-month pilot in high-feedback areas

**Investment:** Moderate (tools, training, contracts)

### Priority 2: Staff Accountability & Scheduling

**Action:** Implement standardized operating hours, attendance tracking, and supervisory spot checks

**Rationale:** 18% of feedback reports closed facilities during operating hours

**Timeline:** 3 months design, 6 months rollout

**Investment:** Low (process redesign, minimal tech)

### Priority 3: Fee Transparency Campaign

**Action:** Publicize which services are free, post fee schedules, establish anonymous reporting channels

**Rationale:** Illegal fee demands erode trust and exclude vulnerable populations

**Timeline:** Immediate (3 months)

**Investment:** Low (signage, hotline, awareness)

### Priority 4: Service Level Agreements (SLAs)

**Action:** Define response times (48-hour acknowledgment, 7-day status update, 30-day resolution target)

**Rationale:** Citizens need predictable responses; SLAs enable performance tracking

**Timeline:** 6 months to pilot

**Investment:** Low to moderate (system adjustments)

### Priority 5: Expand Feedback Reach

**Action:** Partner with community radio, use USSD codes, conduct in-person town halls in underrepresented areas

**Rationale:** Current feedback may miss marginalized groups; proactive outreach needed

**Timeline:** Ongoing (12-month expansion)

**Investment:** Moderate (partnerships, facilitation)

---

## 4. Data & Methodology

### Data Collection
- **Source:** Multi-channel citizen feedback system (SMS, Hotline, Web, In-person, Social Media)
- **Period:** January 2024 – December 2025 (23 months)
- **Volume:** 100 entries (sample dataset for analysis demonstration)

### Analytical Methods
1. **Text Cleaning:** PII masking, normalization, noise removal
2. **Sentiment Analysis:** Rule-based lexicon with negation handling (~75-80% accuracy)
3. **Topic Modeling:** Latent Dirichlet Allocation (LDA) with 5 topics
4. **Descriptive Statistics:** Channel, state, and department distributions

### Limitations
- Small sample size (100 entries) limits generalizability
- Synthetic data used for demonstration; real operational data needed for policy decisions
- English-only analysis; feedback in local languages (Hausa, Yoruba, Igbo, Pidgin) excluded
- Selection bias toward literate, mobile-connected citizens

---

## 5. Conclusion & Next Actions

This analysis demonstrates the value of systematic text analytics for understanding citizen concerns and prioritizing service improvements. The findings—dominated by infrastructure issues, staff absence, and long wait times—align with known challenges in public service delivery but provide quantitative evidence for action.

### Immediate Actions for Donors:
1. **Validate findings** with operational teams and state/LGA administrators
2. **Select pilot sites** (3-5 LGAs) for targeted interventions
3. **Establish baseline metrics** before interventions (e.g., % facilities with working equipment)
4. **Fund rapid response teams** for infrastructure maintenance
5. **Support quarterly reporting** to track progress over time

### Long-Term Vision:
- Real-time sentiment monitoring dashboard for early warning
- Predictive analytics to identify facilities at risk of service breakdown
- Citizen feedback integrated into performance management systems
- Multilingual support to reach all populations
- Open data publication for transparency and accountability

---

## Appendix: Technical Resources

- **Interactive Dashboard:** Real-time visualization at `reporting-outputs/figures/citizen-feedback-dashboard.png`
- **Analysis Notebook:** `notebooks/01_citizen_feedback_eda.ipynb`
- **Source Code:** `src/` directory with full pipeline
- **Documentation:** `docs/` folder with data dictionary, ethics guidelines, modeling notes
- **Data Files:** See `reporting-outputs/annex/citizen-feedback-links.md` for locations

---

**Report Prepared By:** Public Services Analytics Team  
**Contact:** Review project repository for technical details  
**Next Report:** Quarterly updates recommended upon deployment with real data

---

*This analysis uses a sample dataset for demonstration. Findings should be validated with operational data before policy implementation. See ethics guidelines for responsible data handling practices.*
