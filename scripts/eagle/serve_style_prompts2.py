#!/usr/bin/env python3
"""Expanded serving-style prompt set, weighted to the weak styles.

r5 was underfit on the open-ended styles (qa, essay, analysis, code) because the
first prompt set reused ~20 distinct topics per style, repeated 3-5x. The
overfit probe showed the head *can* memorize qa to 0.99 top-1, so the bottleneck
is distinct training data, not capacity. This generator emits a much larger pool
of *distinct* prompts (broad topic lists x style-specific framings), weighted to
the weak styles. Replies are generated with the engine (greedy) by
gen_serving.py, dumped, and used to retrain (r7).

Output: a JSON list of {"style","prompt"} (style is one of
qa/essay/analysis/code/json/math), matching the format gen_serving.py reads.
"""
from __future__ import annotations
import json
import random

random.seed(11)

# Broad topic pool across domains (used to seed style-specific framings).
DOMAINS = [
    # software / systems
    "distributed consensus", "garbage collection", "network packet loss recovery",
    "database index design", "container orchestration", "caching strategies",
    "message queue backpressure", "load balancing", "idempotent retries",
    "query planning", "memory management in C++", "concurrency in Go",
    "the tradeoff between microservices and monoliths", "service mesh overhead",
    "clock synchronization across a cluster", "consistent hashing",
    # science / physics
    "phase transitions", "fluid turbulence", "quantum entanglement",
    "black hole thermodynamics", "protein folding", "neural firing dynamics",
    "ecosystem succession", "stellar nucleosynthesis", "epidemic spread",
    "materials fatigue", "aerosol dispersion", "magnetohydrodynamics",
    # math
    "Markov chains", "gradient descent", "Bayesian inference", "graph colorings",
    "convolution theorem", "central limit theorem", "number partitions",
    "tensor products", "regression diagnostics", "decision trees",
    # society / history / economics
    "monetary policy", "supply chain resilience", "labor market matching",
    "urban sprawl", "voting systems", "public good provision",
    "information cascades", "price formation", "institutional trust",
    "technology diffusion", "demographic transition", "public transit design",
    # biology / medicine
    "immune memory", "circadian rhythm", "gut microbiome", "antibiotic resistance",
    "gene regulation", "aging and telomeres", "neuroplasticity", "drug pharmacokinetics",
    # climate / environment
    "carbon sequestration", "grid storage", "renewable intermittency",
    "ocean circulation", "wildfire behavior", "soil carbon", "atmospheric chemistry",
]

QA_FRAMINGS = [
    "why {t}", "how does {t} work", "what is the core tradeoff in {t}",
    "what are the failure modes of {t}", "why is {t} hard in practice",
    "what do people commonly misunderstand about {t}",
]

ANALYSIS_FRAMINGS = [
    "Analyze step by step: a team notices a symptom related to {t}. List the plausible causes, rank them, and describe the cheapest experiment to distinguish them.",
    "Analyze step by step: two systems using {t} behave differently under the same load. Form hypotheses and the measurement that would confirm each.",
    "Analyze step by step: a metric about {t} looks healthy on average but degrades in the tail. What distributions or mechanisms produce that shape, and how would you verify it?",
    "Analyze step by step: a change to {t} improved a lab metric but hurt a production metric. Reconcile the two signals and give a decision rule.",
]

ESSAY_TOPICS = [
    "the rise and fall of the Suez Canal as a trade choke point",
    "how standardized testing reshaped public education",
    "the economics of open source and its sustainability problem",
    "why cities grew before factories did",
    "the social contract behind international shipping law",
    "how the printing press changed the speed of ideas",
    "the argument for and against universal basic income",
    "why scientific norms reward replication over discovery",
    "the hidden infrastructure of modern payment systems",
    "how urban farming changes a neighborhood's food politics",
    "the tension between patent protection and open research",
    "why some languages are more efficient than others",
    "the case studies in successful rural electrification",
    "how standards bodies decide and who they serve",
    "the sociology of expert testimony in courts",
    "why central planning and market pricing each fail differently",
    "the history of the postal system and why it persists",
    "how measurement uncertainty shapes policy debate",
    "the design of resilient coastal infrastructure",
    "why certain mathematical ideas took centuries to formalize",
]

CODE_TASKS = [
    ("write a function that implements a sliding-window maximum over a stream of integers", "max-window"),
    ("implement an iterator that lazily yields the rows of a 2D jagged array in row-major order", "flatten"),
    ("write a function that parses a duration string like '1h30m12s' into total seconds", "duration"),
    ("implement a function that does a case-insensitive multi-key sort of records with a stable tie-breaker", "sort"),
    ("write a function that computes the longest common subsequence of two strings and returns both length and one alignment", "lcs"),
    ("implement a priority queue that supports decrease-key in O(log n) and returns the smallest element", "pq"),
    ("write a function that detects the start and end of the minimal covering substring containing all given characters", "min-cover"),
    ("implement a function that tokenizes a line of shell input, honoring single and double quotes and backslash escapes", "shelltok"),
    ("write a function that performs a topological sort of a directed graph and reports the first cycle it finds", "tsort"),
    ("implement a thread-safe fixed-size ring buffer with blocking put and non-blocking take", "ring"),
    ("write a function that merges k sorted linked lists into one sorted list", "kmerge"),
    ("implement a function that finds the median of two sorted arrays in O(log n)", "median2"),
    ("write a function that validates a base64 string and returns the decoded bytes, rejecting non-canonical padding", "b64"),
    ("implement a function that groups a list of (timestamp, value) pairs by the hour of the day and returns the per-hour sums", "hour-sum"),
    ("write a function that does a binary search in a rotated sorted array with duplicates", "rot-bisect"),
    ("implement a function that computes the edit distance between two strings with the alignment path", "edit"),
    ("write a function that flattens a nested dict of dicts into dotted-key flat dict, ignoring empty leaves", "flat-dict"),
    ("implement a function that schedules intervals to minimize rooms used and returns the minimum count plus an assignment", "rooms"),
    ("write a function that parses a simple arithmetic expression with + - * / and parentheses into a value", "aritm"),
    ("implement a function that finds the k smallest elements of a list without sorting the whole list", "k-smallest"),
]

JSON_FIELDS = [
    ("name, role, years of experience", "candidate profiles"),
    ("product name, price in cents, currency, in_stock", "product listings"),
    ("date, start_utc, cause, engineers, duration_min", "incident records"),
    ("origin, destination, depart, arrive, gate, on_time", "flight itineraries"),
    ("age, weight_kg, height_m, bp, heart_rate", "patient vitals"),
    ("order_id, item names with qty and price, total, payment", "order summaries"),
    ("host, cpu_pct, mem_pct, disk_io, region", "server telemetry"),
    ("title, author, year, venue, citations", "paper metadata"),
    ("account, tx_type, amount, fee, balance_after", "bank transactions"),
    ("sensor_id, metric, min, max, unit, sampled_at", "sensor readings"),
    ("city, population, elevation, timezone", "city profiles"),
    ("repo, stars, forks, license, last_push", "repository stats"),
    ("patient, diagnosis, severity, onset_date, treating_physician", "clinic notes"),
    ("match, home_score, away_score, duration_min, venue", "sports results"),
    ("course, instructor, credits, schedule, capacity", "course catalog"),
    ("incident, severity, affected_services, root_cause, mitigation", "ops runbooks"),
]


JSON_RECORDS = [
    ("Extract as JSON only, no prose, the fields name, role, and years of experience from: 'Interview notes: our candidate Marta Kovac works as a lead backend engineer at a logistics company, with 9 years of professional experience after 2 more as a contractor.'",
     '{"name": "Marta Kovac", "role": "lead backend engineer", "years": 9}'),
    ("Extract as JSON only the product name, price in cents, currency, and in_stock from: 'The Nova Desk Lamp, model ND-200, is listed at $49.99 USD in the spring catalog and is in stock.'",
     '{"name": "Nova Desk Lamp", "price_cents": 4999, "currency": "USD", "in_stock": true}'),
    ("Extract as JSON only the fields from: 'The incident on 2026-11-01 started at 14:23 UTC, was caused by a bad deployment, and took 3 engineers and 47 minutes to resolve.'",
     '{"date": "2026-11-01", "start": "14:23", "cause": "bad deployment", "engineers": 3, "duration_min": 47}'),
    ("Extract as JSON only the fields from: 'The flight AA1042 from JFK to LAX departs 07:45, arrives 10:50, gate B22, and is on time.'",
     '{"flight": "AA1042", "from": "JFK", "to": "LAX", "depart": "07:45", "arrive": "10:50", "gate": "B22", "on_time": true}'),
    ("Extract as JSON only the fields from: 'Patient record: 34-year-old, 72 kg, height 1.78 m, blood pressure 121/78, heart rate 61 bpm at rest.'",
     '{"age": 34, "weight_kg": 72, "height_m": 1.78, "bp_systolic": 121, "bp_diastolic": 78, "hr": 61}'),
    ("Extract as JSON only the fields from: 'The order #A7712 contains 2x USB-C hub at $18.50 each and 1x cable at $9.00, total $46.00, paid by card.'",
     '{"order": "A7712", "items": [{"name": "USB-C hub", "qty": 2, "price": 18.5}, {"name": "cable", "qty": 1, "price": 9.0}], "total": 46.0, "payment": "card"}'),
    ("Extract as JSON only the fields from: 'Server db-03 in eu-west-1 reports 82% CPU, 61% memory, 0.4 disk io, and is healthy.'",
     '{"host": "db-03", "cpu_pct": 82, "mem_pct": 61, "disk_io": 0.4, "region": "eu-west-1", "healthy": true}'),
    ("Extract as JSON only the fields from: 'The paper Attention Is All You Need was written by Vaswani et al., published in 2017 at NIPS, and has been cited 128400 times.'",
     '{"title": "Attention Is All You Need", "author": "Vaswani et al.", "year": 2017, "venue": "NIPS", "citations": 128400}'),
    ("Extract as JSON only the fields from: 'Transaction from checking: a transfer of $350.00, fee $0.00, leaving a balance of $1,240.50.'",
     '{"account": "checking", "tx_type": "transfer", "amount": 350.0, "fee": 0.0, "balance_after": 1240.5}'),
    ("Extract as JSON only the fields from: 'Sensor T-118 logged temperature in Celsius, min 4.2, max 21.9, at 2026-10-30T08:15Z.'",
     '{"sensor_id": "T-118", "metric": "temperature", "min": 4.2, "max": 21.9, "unit": "C", "sampled_at": "2026-10-30T08:15Z"}'),
    ("Extract as JSON only the fields from: 'The city of Bergen, Norway, has a population of 298000, an elevation of 12 m, and is in the Europe/Oslo timezone.'",
     '{"city": "Bergen", "population": 298000, "elevation": 12, "timezone": "Europe/Oslo"}'),
    ("Extract as JSON only the fields from: 'The repo granite4 has 4100 stars, 380 forks, a MIT license, and its last push was 2026-11-02.'",
     '{"repo": "granite4", "stars": 4100, "forks": 380, "license": "MIT", "last_push": "2026-11-02"}'),
    ("Extract as JSON only the fields from: 'Clinic note: patient J. Osei, diagnosis mild asthma, severity 2, onset 2026-09-14, treating physician Dr. Lindqvist.'",
     '{"patient": "J. Osei", "diagnosis": "mild asthma", "severity": 2, "onset_date": "2026-09-14", "treating_physician": "Dr. Lindqvist"}'),
    ("Extract as JSON only the fields from: 'Match: Arsenal 2, Chelsea 1, after 95 minutes, at the Emirates.'",
     '{"match": "Arsenal vs Chelsea", "home_score": 2, "away_score": 1, "duration_min": 95, "venue": "the Emirates"}'),
    ("Extract as JSON only the fields from: 'Course: Linear Algebra, taught by Prof. Okafor, worth 5 credits, meets Tue/Thu 10:00, capacity 120.'",
     '{"course": "Linear Algebra", "instructor": "Prof. Okafor", "credits": 5, "schedule": "Tue/Thu 10:00", "capacity": 120}'),
    ("Extract as JSON only the fields from: 'Ops incident: severity 2, affected services are search and auth, root cause was a bad cache flush, mitigation was to restart the cache tier.'",
     '{"incident": "sev2-search-auth", "severity": 2, "affected_services": ["search", "auth"], "root_cause": "bad cache flush", "mitigation": "restart the cache tier"}'),
]


def _pick(seq, k):
    return random.sample(seq, min(k, len(seq)))


def _cycle(pool, n):
    """Repeat a list cyclically to exactly n items (preserves distinct order)."""
    return [pool[i % len(pool)] for i in range(n)]


def main():
    # Target the serving distribution (essay 30%, qa/analysis/code/math 15%,
    # json 10%) at 480 prompts — a bit more than the original 400. The weak
    # styles get far more *distinct* coverage than the original set (qa now from
    # a 360-distinct pool, analysis from 180, vs ~20/~15 before).
    targets = {"essay": 144, "qa": 72, "analysis": 72, "code": 72, "math": 72, "json": 48}

    # qa: 360 distinct -> sample 72
    qa_topics = _pick(DOMAINS, 60)
    qa_pool = [f"Answer in a few sentences: {f.format(t=t)}?"
               for f in QA_FRAMINGS for t in qa_topics]
    # analysis: 180 distinct -> sample 72
    an_topics = _pick(DOMAINS, 45)
    an_pool = [f.format(t=t) for f in ANALYSIS_FRAMINGS for t in an_topics]
    # essay: 40 distinct (20 topics x 2 framings)
    essay_pool = []
    for t in ESSAY_TOPICS:
        essay_pool.append(f"Write a 250-word essay about {t}")
        essay_pool.append(f"Explain, in three short paragraphs, {t}")
    # code: 20 distinct
    code_pool = [task for task, _ in CODE_TASKS]
    # json: 16 distinct
    json_pool = [p for p, _ in JSON_RECORDS]
    # math: 20 distinct
    math_pool = [
        "Solve step by step: a train leaves A at 9:00 at 60 km/h; another leaves B at 9:30 at 80 km/h toward A. The distance between A and B is 280 km. When do they meet and where?",
        "Solve step by step: the sum of three consecutive multiples of 5 is 405. What are they?",
        "Solve step by step: a rectangle's diagonal is 13 and its area is 30. What is its perimeter?",
        "Solve step by step: 15% of x plus 8 is equal to 27. Find x.",
        "Solve step by step: how many ways can you choose 3 items from 10 without order mattering?",
        "Solve step by step: two pipes fill a tank in 4 and 6 hours; a drain empties it in 10 hours. With all three open, how long to fill the tank?",
        "Solve step by step: find the value of 2^10 - 10^2 divided by 4.",
        "Solve step by step: the ratio of boys to girls is 3:4 and there are 35 students. How many girls?",
        "Solve step by step: a circular track is 400 m. Two runners start together going opposite ways at 6 and 9 m/s. How often do they meet in one hour?",
        "Solve step by step: if a is 30% more than b, by what percent is b less than a?",
        "Solve step by step: a ladder 5 m long leans against a wall with its base 1.4 m from the wall. How high up the wall does it reach?",
        "Solve step by step: the average of five numbers is 40. If the largest is removed the average is 32. What was the largest?",
        "Solve step by step: what is 12% of 85 plus 5% of 400?",
        "Solve step by step: a number divided by 7 gives a remainder of 3; divided by 5 it gives a remainder of 2. Find the smallest such number.",
        "Solve step by step: a clock's hour and minute hands are 120 degrees apart between 2 and 3 o'clock. What time is it?",
        "Solve step by step: a shop marks up a product by 40% then gives a 20% discount. What is the net effect on the original price as a percent?",
        "Solve step by step: five workers finish a job in 8 days; two leave after 4 days. How many more days do the remaining workers need?",
        "Solve step by step: a boat travels 12 km downstream in 2 h and back in 3 h. Find the boat's speed in still water and the current's speed.",
        "Solve step by step: the ratio of two numbers is 3:5 and their difference is 24. Find the numbers.",
        "Solve step by step: a farmer has chickens and rabbits. There are 30 heads and 88 legs. How many of each?",
    ]
    pools = {"qa": qa_pool, "analysis": an_pool, "essay": essay_pool,
             "code": code_pool, "json": json_pool, "math": math_pool}

    out = []
    for style in ("essay", "qa", "analysis", "code", "math", "json"):
        for p in _cycle(pools[style], targets[style]):
            out.append({"style": style, "prompt": p})

    import collections, sys
    c = collections.Counter(x["style"] for x in out)
    print(f"{len(out)} prompts: {dict(c)}", file=sys.stderr, flush=True)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
