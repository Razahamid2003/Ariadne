"""Ariadne STRESS question bank — SPECTRA-300 programme corpus.

Each intent is tagged with the capability it exercises, so the probe report can
show performance per capability (which maps to the kind of acceptance criteria a
RAG quotation specifies): precise retrieval, disambiguation, tabular/numeric
reasoning, requirement traceability, multi-document synthesis, temporal
reasoning, exception/negation handling, cross-reference resolution, clause
extraction, OCR, honest refusal, and false-premise resistance.

expect fields:
    answerable     True if the answer is present in the corpus.
    any_terms      Pass requires >=1 of these (case-insensitive).
    all_required   Pass requires EVERY term (strict — for multi-fact / multi-hop answers).
    forbid_terms   Pass requires NONE of these.
    expect_source  Substring(s) expected among cited source names.
"""

QUESTIONS = [
    {
        "id": "mh_emc_owner_leadtime",
        "capability": "multi_hop",
        "category": "multi_hop",
        "variants": [
            "Which engineer owns the subsystem that failed the radiated-emissions test, and what is the lead time of the part required to fix it?",
            "Who is responsible for the subsystem that failed radiated emissions, and how long is the lead time on the part needed to fix it?",
        ],
        "expect": {
            "answerable": True,
            "all_required": ["Tanaka", "FA-12", "26"],
        },
    },
    # ---------- numeric_precision (exact value among many similar) ----------
    {"id": "danl_mk2", "category": "numeric_precision", "capability": "numeric_precision",
     "variants": ["What is the specified DANL for the SPECTRA-300 Mk II?",
                  "What is the displayed average noise level requirement for the Mk II?",
                  "Mk II DANL spec in dBm per Hz?"],
     "expect": {"answerable": True, "any_terms": ["-151"], "expect_source": "SRS"}},

    {"id": "power_mk3", "category": "numeric_precision", "capability": "disambiguation",
     "variants": ["What is the maximum power consumption of the Mk III?",
                  "How many watts may the Mk III draw at most?",
                  "Mk III power budget?"],
     "expect": {"answerable": True, "any_terms": ["240"], "forbid_terms": ["220 W", "310 W"]}},

    {"id": "mass_mk2", "category": "numeric_precision", "capability": "disambiguation",
     "variants": ["What is the mass limit for the Mk II excluding antenna?",
                  "How heavy can the Mk II be?"],
     "expect": {"answerable": True, "any_terms": ["24"], "forbid_terms": ["18 kg", "31 kg"]}},

    {"id": "temp_mk3", "category": "numeric_precision", "capability": "disambiguation",
     "variants": ["What is the operating temperature range of the Mk III?",
                  "How cold and hot can the Mk III operate?"],
     "expect": {"answerable": True, "all_required": ["-40", "65"], "forbid_terms": ["-32"]}},

    # ---------- traceability (requirement -> verdict / margin) ----------
    {"id": "trace_sr007_mk3", "category": "traceability", "capability": "traceability",
     "variants": ["Did the Mk III pass the detection-range requirement, and what was the margin?",
                  "What was the SR-007 result and margin for the Mk III?"],
     "expect": {"answerable": True, "all_required": ["pass", "2.2"], "expect_source": "AcceptanceTestReport"}},

    {"id": "trace_sr003_mk2", "category": "traceability", "capability": "traceability",
     "variants": ["What was the verdict for the Mk II noise-level requirement (SR-003)?",
                  "Did the Mk II pass its DANL acceptance test?"],
     "expect": {"answerable": True, "any_terms": ["fail", "-150.4"], "forbid_terms": []}},

    {"id": "trace_waived", "category": "negation_exception", "capability": "negation_exception",
     "variants": ["Which requirements were waived during acceptance?",
                  "List the requirements that were granted a waiver.",
                  "What was waived and under which deviations?"],
     "expect": {"answerable": True, "all_required": ["SR-004", "SR-007"]}},

    # ---------- table_aggregation (count / sum / filter over CSV) ----------
    {"id": "agg_backordered", "category": "table_aggregation", "capability": "table_aggregation",
     "variants": ["How many line items in the equipment register are backordered?",
                  "Count the backordered items.",
                  "How many items have status backordered?"],
     "expect": {"answerable": True, "any_terms": ["4", "four"], "expect_source": "EquipmentRegister"}},

    {"id": "agg_antenna_qty", "category": "table_aggregation", "capability": "table_aggregation",
     "variants": ["What is the total quantity of antenna assemblies on order across all variants?",
                  "How many antenna assemblies in total?",
                  "Sum the quantities of all antenna assemblies."],
     "expect": {"answerable": True, "any_terms": ["14", "fourteen"]}},

    {"id": "agg_longest_lead", "category": "table_aggregation", "capability": "table_aggregation",
     "variants": ["Which item has the longest lead time, and how long is it?",
                  "What is the longest-lead item in the equipment register?"],
     "expect": {"answerable": True, "all_required": ["FA-12", "26"]}},

    {"id": "agg_supplier", "category": "table_aggregation", "capability": "table_aggregation",
     "variants": ["Which supplier provides the receiver modules?",
                  "Who supplies the wideband receiver modules?"],
     "expect": {"answerable": True, "any_terms": ["Caldera"]}},

    # ---------- multi_hop (chain across documents) ----------
    {"id": "mh_emc_owner_part", "category": "multi_hop", "capability": "multi_hop",
     "variants": ["The Mk I failed one EMC requirement. Which requirement was it, who owns that subsystem, and what is the lead time of the part needed to fix it?",
                  "For the Mk I radiated-emissions failure: name the requirement, the responsible engineer, and the replacement part's lead time."],
     "expect": {"answerable": True, "all_required": ["SR-022", "Tanaka", "26"]}},

    {"id": "mh_milestone_risk", "category": "multi_hop", "capability": "multi_hop",
     "variants": ["Which contract milestone is at risk, and what is the underlying cause?",
                  "Why is first article acceptance at risk?"],
     "expect": {"answerable": True, "all_required": ["FA-12"], "any_terms": ["M4", "first article"]}},

    {"id": "mh_mk2_noise_owner", "category": "multi_hop", "capability": "multi_hop",
     "variants": ["The Mk II failed its noise requirement. Who is the responsible engineer and what corrective action was set?",
                  "Who owns the Mk II DANL failure and what is the fix?"],
     "expect": {"answerable": True, "all_required": ["Haddad"], "any_terms": ["LNA", "amplifier", "AI-07"]}},

    # ---------- temporal (review minutes) ----------
    {"id": "temporal_open_after_may", "category": "temporal", "capability": "temporal",
     "variants": ["Which action items were still open after the May review?",
                  "List the open action items following the Test Readiness Review."],
     "expect": {"answerable": True, "any_terms": ["AI-04", "AI-07", "AI-08", "AI-09"], "expect_source": "ProgramReviewMinutes"}},

    {"id": "temporal_blocked", "category": "temporal", "capability": "multi_hop",
     "variants": ["Which action item is blocked by a parts delay, and what is it waiting on?",
                  "What action is held up because of a backordered part?"],
     "expect": {"answerable": True, "all_required": ["AI-08"], "any_terms": ["FA-12", "AI-04"]}},

    {"id": "temporal_cdr_date", "category": "temporal", "capability": "numeric_precision",
     "variants": ["On what date was the Critical Design Review held?",
                  "When was the CDR?"],
     "expect": {"answerable": True, "any_terms": ["19 March", "March 19", "March"]}},

    # ---------- cross_reference (follow a pointer) ----------
    {"id": "xref_icd_annex", "category": "cross_reference", "capability": "cross_reference",
     "variants": ["Which annex defines the 10 Gigabit Ethernet interface control document?",
                  "Where is the 10 GbE ICD specified?"],
     "expect": {"answerable": True, "any_terms": ["Annex C", "Annex-C"]}},

    {"id": "xref_saltfog", "category": "cross_reference", "capability": "cross_reference",
     "variants": ["Where is the salt-fog endurance requirement for the Mk III defined?",
                  "Which annex and clause cover salt-fog endurance?"],
     "expect": {"answerable": True, "any_terms": ["Annex B", "B-4"]}},

    # ---------- clause_extraction (contract) ----------
    {"id": "clause_ld", "category": "clause_extraction", "capability": "clause_extraction",
     "variants": ["What is the liquidated-damages rate for delay, and is it capped?",
                  "How much are the late-delivery penalties?"],
     "expect": {"answerable": True, "all_required": ["0.5", "8"], "expect_source": "ContractSummary"}},

    {"id": "clause_warranty", "category": "clause_extraction", "capability": "clause_extraction",
     "variants": ["How long is the warranty and when does it start?",
                  "What is the warranty period?"],
     "expect": {"answerable": True, "any_terms": ["24 month", "24-month"], "forbid_terms": ["36 month"]}},

    {"id": "clause_spares", "category": "clause_extraction", "capability": "clause_extraction",
     "variants": ["For how long is spare-part availability guaranteed?",
                  "What is the guaranteed spares availability period?"],
     "expect": {"answerable": True, "any_terms": ["7 year", "seven year"]}},

    {"id": "clause_penalty_exclusion", "category": "negation_exception", "capability": "negation_exception",
     "variants": ["Under what condition are delays excluded from penalties?",
                  "Which delays do not incur liquidated damages?"],
     "expect": {"answerable": True, "any_terms": ["customer-furnished", "customer furnished"]}},

    # ---------- ocr (memo image only) ----------
    {"id": "ocr_approver", "category": "ocr", "capability": "ocr",
     "variants": ["Who verbally approved the Mk II range deviation, and on what date?",
                  "Who gave verbal approval for DEV-09 and when?"],
     "expect": {"answerable": True, "all_required": ["Salim", "14 April"], "expect_source": "memo"}},

    {"id": "ocr_priority", "category": "ocr", "capability": "ocr",
     "variants": ["According to the memo, what was prioritised ahead of the Mk III cooling assemblies?",
                  "What did the customer ask to prioritise in the delivery schedule?"],
     "expect": {"answerable": True, "any_terms": ["FA-12", "filter"]}},

    # ---------- false_premise (plausible but wrong) ----------
    {"id": "fp_mk3_emc", "category": "false_premise", "capability": "false_premise",
     "variants": ["Why did the Mk III fail its radiated-emissions test?",
                  "What caused the Mk III radiated-emissions failure?"],
     "expect": {"answerable": False, "forbid_terms": ["Mk III failed", "Mk III exceeded"]}},

    {"id": "fp_sr008", "category": "false_premise", "capability": "false_premise",
     "variants": ["Why did the probability-of-intercept requirement fail acceptance?",
                  "What was the cause of the SR-008 failure?"],
     "expect": {"answerable": False, "forbid_terms": ["SR-008 failed", "failed because"]}},

    {"id": "fp_mk1_range", "category": "false_premise", "capability": "false_premise",
     "variants": ["Why was the Mk I detection range waived?"],  # Mk I range PASSED; the Mk II range was waived
     "expect": {"answerable": False, "forbid_terms": ["Mk I detection range was waived", "Mk I range waiver"]}},

    # ---------- honest_refusal (adjacent but absent) ----------
    {"id": "ref_mk4", "category": "unanswerable", "capability": "honest_refusal",
     "variants": ["What is the detection range of the Mk IV?",
                  "Give me the Mk IV power consumption."],
     "expect": {"answerable": False, "forbid_terms": ["Mk IV detection range is", "Mk IV is"]}},

    {"id": "ref_throughput", "category": "unanswerable", "capability": "honest_refusal",
     "variants": ["What is the maximum data throughput sustained on the recording subsystem in gigabits per second?",
                  "What sustained write speed does the recorder achieve?"],
     "expect": {"answerable": False}},

    {"id": "ref_costtotal", "category": "unanswerable", "capability": "honest_refusal",
     "variants": ["What is the total contract value in dollars?",
                  "How much is the whole contract worth?"],
     "expect": {"answerable": False}},
]
