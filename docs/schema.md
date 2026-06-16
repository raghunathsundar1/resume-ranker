# Candidate schema (verified against sample_candidates.json)

Field-name mapping lives ONLY in `normalize.py` (`FIELD_MAP`, `PROFILE_MAP`,
`EMP_MAP`, `SKILL_MAP`). If the full pool reveals new names, edit those tables and
nothing else. This file is the authoritative description; do not duplicate it in code.

## Record shape

```
candidate_id : str                      # "CAND_0000001"
profile : {
  anonymized_name, headline, summary, location, country,
  years_of_experience : float,          # self-reported total
  current_title, current_company,
  current_company_size,                 # e.g. "10001+"
  current_industry                      # e.g. "IT Services"  <- JD-critical
}
career_history : [ {
  company, title, start_date, end_date, # ISO "YYYY-MM-DD"; end_date null if current
  duration_months : int,                # AUTHORITATIVE tenure source (not the dates)
  is_current : bool, industry, company_size, description
} ]
education : [ { institution, degree, field_of_study,
                start_year, end_year, grade, tier } ]   # tier: "tier_1".."tier_5"
skills : [ { name, proficiency, endorsements, duration_months } ]
          # proficiency in {beginner, intermediate, advanced, expert}
certifications : [...]    languages : [...]
redrob_signals : { ...23 keys... }       # see below
```

## redrob_signals (23 keys) — the behavioral-twin separators

`profile_completeness_score, signup_date, last_active_date, open_to_work_flag,
profile_views_received_30d, applications_submitted_30d, recruiter_response_rate,
avg_response_time_hours, skill_assessment_scores{dict}, connection_count,
endorsements_received, notice_period_days, expected_salary_range_inr_lpa{min,max},
preferred_work_mode, willing_to_relocate, github_activity_score,
search_appearance_30d, saved_by_recruiters_30d, interview_completion_rate,
offer_acceptance_rate, verified_email, verified_phone, linkedin_connected`

`skill_assessment_scores` (e.g. an NLP score) is objective corroboration for claimed
skills — weight it in scoring. `normalize.py` flattens all signals to `sig_*` columns
and expands the two nested dicts into `_min/_max` and `_max/_mean` features.

## Sample-data realities (50-record sample)

- 36/50 in India; the rest abroad. JD prefers Noida/Pune, India in-scope.
- 20/50 currently in "IT Services" — directly drives the anti-consulting /
  pro-product-company logic (`is_services_now`, `services_years`, `product_years`).
- Sample is essentially honeypot-free (~80 in 100K → ~0 in 50). Do not "fix" the
  integrity gate because it fires rarely on the sample.