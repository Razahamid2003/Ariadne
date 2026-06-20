"""Ariadne evaluation question bank.

Each entry is one *intent* with several phrasings (``variants``) so retrieval is
tested for paraphrase robustness, abbreviations, and typos. ``expect`` records
what a correct answer must (and must not) contain, so the probe can grade
responses automatically.

expect fields:
    answerable     True if the answer is present in the test corpus.
    any_terms      Pass requires at least one of these (case-insensitive) in the answer.
    all_terms      Pass requires ALL of these in the answer (optional).
    forbid_terms   Pass requires NONE of these in the answer (optional).
    expect_source  Substring expected among the cited source document names (optional).

Categories: factual, numeric, date, list_aggregation, cross_document,
paraphrase_robust, abbrev_typo, doctype_targeted, unanswerable,
adversarial_false_premise.
"""

QUESTIONS = [
    # ---- Simple factual ----
    {"id": "ceo", "category": "factual",
     "variants": ["Who is the CEO of Helios Dynamics?", "Who runs Helios Dynamics?",
                  "Name the chief executive of the company.", "whos the boss at helios"],
     "expect": {"answerable": True, "any_terms": ["Marion Vale", "Vale"],
                "expect_source": "company_handbook"}},

    {"id": "cto", "category": "factual",
     "variants": ["Who is the CTO?", "Who is the chief technology officer of Helios?",
                  "Name the head of technology."],
     "expect": {"answerable": True, "any_terms": ["Anil Raghavan", "Raghavan"]}},

    {"id": "hq", "category": "factual",
     "variants": ["Where is Helios Dynamics headquartered?", "What city is the company based in?",
                  "Where is the head office located?", "where is helios HQ"],
     "expect": {"answerable": True, "any_terms": ["Austin"], "expect_source": "company_handbook"}},

    # ---- Dates / founding ----
    {"id": "founded", "category": "date",
     "variants": ["When was Helios Dynamics founded?", "What year did the company start?",
                  "How old is Helios Dynamics?", "founding year of helios"],
     "expect": {"answerable": True, "any_terms": ["2014"]}},

    # ---- Numeric facts ----
    {"id": "headcount", "category": "numeric",
     "variants": ["How many employees does Helios Dynamics have?", "What is the company headcount?",
                  "How many people work at Helios?"],
     "expect": {"answerable": True, "any_terms": ["480"]}},

    {"id": "pto", "category": "numeric",
     "variants": ["How many PTO days do full-time employees get?",
                  "How much paid time off do staff receive each year?",
                  "How many vacation days are there?", "annual PTO allowance"],
     "expect": {"answerable": True, "any_terms": ["22"], "expect_source": "company_handbook"}},

    {"id": "remote_days", "category": "numeric",
     "variants": ["How many days a week can employees work remotely?",
                  "What is the remote work limit per week?", "how often can I work from home"],
     "expect": {"answerable": True, "any_terms": ["3", "three"]}},

    # ---- Precise spec retrieval (Atlas-7) ----
    {"id": "atlas_mass", "category": "numeric",
     "variants": ["What is the mass of the Atlas-7?", "How much does the Atlas-7 weigh?",
                  "Atlas-7 weight?", "what's the wieght of the atlas 7"],  # typo intentional
     "expect": {"answerable": True, "any_terms": ["3.2"], "expect_source": "atlas"}},

    {"id": "atlas_range", "category": "numeric",
     "variants": ["What is the detection range of the Atlas-7?", "How far can the Atlas-7 detect?",
                  "Atlas-7 range in km?"],
     "expect": {"answerable": True, "any_terms": ["40"], "expect_source": "atlas"}},

    {"id": "atlas_power", "category": "numeric",
     "variants": ["What is the power consumption of the Atlas-7?", "How many watts does the Atlas-7 draw?",
                  "Atlas-7 power draw?"],
     "expect": {"answerable": True, "any_terms": ["28"]}},

    {"id": "atlas_mtbf", "category": "abbrev_typo",
     "variants": ["What is the MTBF of the Atlas-7?",
                  "What is the mean time between failures for the Atlas-7?",
                  "How reliable is the Atlas-7 in hours?"],
     "expect": {"answerable": True, "any_terms": ["12,000", "12000"], "expect_source": "atlas"}},

    {"id": "atlas_temp", "category": "numeric",
     "variants": ["What is the operating temperature range of the Atlas-7?",
                  "How cold can the Atlas-7 operate?", "Atlas-7 temperature limits?"],
     "expect": {"answerable": True, "any_terms": ["-40", "65"]}},

    {"id": "atlas_warranty", "category": "numeric",
     "variants": ["What is the warranty on the Atlas-7?", "How long is the Atlas-7 guaranteed for?",
                  "atlas 7 warranty period"],
     "expect": {"answerable": True, "any_terms": ["36"], "expect_source": ["atlas", "faq"]}},

    # ---- Borealis-3 ----
    {"id": "borealis_mass", "category": "numeric",
     "variants": ["What is the mass of the Borealis-3?", "How much does the Borealis-3 weigh?"],
     "expect": {"answerable": True, "any_terms": ["1.1"], "expect_source": "borealis"}},

    {"id": "borealis_warranty", "category": "numeric",
     "variants": ["What is the warranty on the Borealis-3?", "Borealis-3 warranty length?"],
     "expect": {"answerable": True, "any_terms": ["24"]}},

    # ---- Cross-document reasoning ----
    {"id": "compare_range", "category": "cross_document",
     "variants": ["Which has a longer detection range, the Atlas-7 or the Borealis-3?",
                  "Compare the range of the Atlas-7 and Borealis-3.",
                  "Which sensor sees farther, Atlas or Borealis?"],
     "expect": {"answerable": True, "any_terms": ["Atlas"], "all_terms": ["40"], }},

    {"id": "compare_mtbf", "category": "cross_document",
     "variants": ["Which product has the higher MTBF, Atlas-7 or Borealis-3?",
                  "Which is more reliable by MTBF, the Atlas-7 or the Borealis-3?"],
     "expect": {"answerable": True, "any_terms": ["Borealis"], "all_terms": ["18,000", "18000"]}},

    # ---- Security policy ----
    {"id": "password_length", "category": "numeric",
     "variants": ["What is the minimum password length?", "How many characters must a password have?",
                  "minimum password size policy"],
     "expect": {"answerable": True, "any_terms": ["14"], "expect_source": "security"}},

    {"id": "password_rotation", "category": "numeric",
     "variants": ["How often must passwords be changed?", "What is the password rotation period?",
                  "how frequently do I change my password"],
     "expect": {"answerable": True, "any_terms": ["90"]}},

    {"id": "data_retention", "category": "numeric",
     "variants": ["How long are customer records retained?", "What is the data retention period?",
                  "how many years do we keep customer data"],
     "expect": {"answerable": True, "any_terms": ["7", "seven"], "expect_source": "security"}},

    {"id": "incident_time", "category": "numeric",
     "variants": ["How quickly must a suspected breach be reported?",
                  "What is the breach reporting deadline?", "incident reporting time limit"],
     "expect": {"answerable": True, "any_terms": ["1 hour", "one hour", "within 1"]}},

    # ---- Procedures ----
    {"id": "rma", "category": "factual",
     "variants": ["How do I request a repair?", "What is the RMA process?",
                  "How do I send hardware back for repair?"],
     "expect": {"answerable": True, "any_terms": ["RMA", "support@heliosdynamics"],
                "forbid_terms": [], "expect_source": "faq"}},

    {"id": "support_response", "category": "numeric",
     "variants": ["What is the standard support response time?", "How fast does support reply?",
                  "priority support response time"],
     "expect": {"answerable": True, "any_terms": ["8", "2", "business hours"]}},

    # ---- Report figures ----
    {"id": "q3_revenue", "category": "numeric",
     "variants": ["What was Q3 revenue?", "How much revenue did the company make in Q3?",
                  "quarterly revenue figure"],
     "expect": {"answerable": True, "any_terms": ["14.2"], "expect_source": "quarterly"}},

    {"id": "q3_growth", "category": "numeric",
     "variants": ["What was the year-over-year growth in Q3?", "How much did revenue grow?",
                  "YoY growth percentage"],
     "expect": {"answerable": True, "any_terms": ["18"]}},

    {"id": "q3_region", "category": "factual",
     "variants": ["Which region contributed the most revenue in Q3?", "What was the strongest region?",
                  "top region by revenue"],
     "expect": {"answerable": True, "any_terms": ["North America"]}},

    # ---- Document-type targeted (resume) ----
    {"id": "resume_experience", "category": "doctype_targeted",
     "variants": ["How many years of experience does Layla Haddad have?",
                  "What is Layla Haddad's experience level?", "Haddad years of experience"],
     "expect": {"answerable": True, "any_terms": ["9", "nine"], "expect_source": "haddad"}},

    {"id": "resume_education", "category": "doctype_targeted",
     "variants": ["Where did Layla Haddad study?", "What is Haddad's educational background?",
                  "which universities did Layla Haddad attend"],
     "expect": {"answerable": True, "any_terms": ["Toronto", "McGill"]}},

    # ---- Table / list aggregation ----
    {"id": "dir_engineering", "category": "list_aggregation",
     "variants": ["Who works in the Engineering department?", "List the engineering staff.",
                  "which employees are in engineering"],
     "expect": {"answerable": True, "any_terms": ["Raghavan", "Haddad", "Lindqvist"]}},

    {"id": "dir_denver", "category": "list_aggregation",
     "variants": ["Who is based in Denver?", "List employees in the Denver office.",
                  "which staff are in denver"],
     "expect": {"answerable": True, "any_terms": ["Reyes", "Okonkwo"]}},

    {"id": "dir_l4", "category": "list_aggregation",
     "variants": ["Which employees have L4 clearance?", "Who holds the highest clearance level?",
                  "list L4 clearance holders"],
     "expect": {"answerable": True, "any_terms": ["Vale", "Raghavan"]}},

    # ---- Unanswerable (must decline honestly) ----
    {"id": "unanswerable_product", "category": "unanswerable",
     "variants": ["What is the detection range of the Helios Titan-9?",
                  "Give me the specs of the Titan-9 sensor.", "How much does the Titan-9 weigh?"],
     "expect": {"answerable": False, "forbid_terms": ["Titan-9 is", "km", "kg"]}},

    {"id": "unanswerable_dividend", "category": "unanswerable",
     "variants": ["What is the company's dividend policy?", "How much dividend does Helios pay?",
                  "when is the next dividend payment"],
     "expect": {"answerable": False}},

    {"id": "unanswerable_ceo_salary", "category": "unanswerable",
     "variants": ["What is the CEO's salary?", "How much is Marion Vale paid?"],
     "expect": {"answerable": False}},

    # ---- Adversarial false premise (must not agree / hallucinate) ----
    {"id": "adv_relocate", "category": "adversarial_false_premise",
     "variants": ["Why did Helios Dynamics relocate its headquarters to Seattle?",
                  "When did the company move to Seattle?"],
     "expect": {"answerable": False, "forbid_terms": ["Seattle"]}},

    {"id": "adv_false_ceo", "category": "adversarial_false_premise",
     "variants": ["Why did CEO John Smith resign?", "When did John Smith step down as CEO?"],
     "expect": {"answerable": False, "forbid_terms": ["John Smith resigned", "John Smith stepped"]}},

    {"id": "adv_bankruptcy", "category": "adversarial_false_premise",
     "variants": ["Why did Helios Dynamics file for bankruptcy in 2020?",
                  "What caused the 2020 Helios bankruptcy?"],
     "expect": {"answerable": False, "forbid_terms": ["bankruptcy was", "filed for bankruptcy in 2020 because"]}},
]
