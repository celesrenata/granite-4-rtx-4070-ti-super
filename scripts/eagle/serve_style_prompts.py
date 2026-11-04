#!/usr/bin/env python3
"""Generate serving-style training prompts for EAGLE head training.

The chat-only training set leaves the head underfit on the structured styles the
service actually serves (JSON extraction, code edits, step-by-step analysis,
math, essays). This emits ~500 diverse single-turn prompts in those styles; the
8B replies are generated with the engine (greedy - identical to what it serves
with EAGLE), then dumped with `dump-hidden` and trained like selfgen24.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

random.seed(7)

TOPICS = [
    "distributed systems", "machine learning interpretability",
    "the history of the Roman Empire", "climate adaptation in cities",
    "protein folding", "the economics of public transit",
    "quantum error correction", "gardening in cold climates",
    "the philosophy of free will", "supply chain resilience",
    "the history of jazz", "ocean currents",
    "the psychology of procrastination", "ancient trade routes",
    "the ethics of synthetic data", "urban agriculture", "compiler design",
    "the biology of sleep", "the history of paper money",
    "wildlife corridors", "the sociology of open source", "tidal energy",
    "the mathematics of voting systems", "beekeeping",
    "the history of lighthouses", "soil microbiome",
    "the physics of supersonic flight", "the history of the printing press",
    "wildfire smoke and air quality", "the ethics of facial recognition"]

QA_TOPICS = [
    "why do deep learning models need batch normalization",
    "how does a B-tree differ from a skip list",
    "what causes jet lag and why is it worse eastbound",
    "why do black holes have event horizons",
    "how do vaccines train the immune system without causing disease",
    "why is UTF-8 more common than UTF-16",
    "what is the tradeoff between recall and precision",
    "how do GPUs use tensor cores for matrix multiplication",
    "why do rivers form meanders",
    "how does RAID 5 differ from RAID 6",
    "why does salt melt ice on roads",
    "how do neural radiance fields represent 3D scenes",
    "what is the difference between a monad and a functor",
    "why do magnets attract",
    "how does TCP avoid duplicate packets flooding a receiver",
    "what causes auroras",
    "why do we use logarithmic scales for sound",
    "how do GPS satellites account for relativity",
    "what is the purpose of a residual connection",
    "why do crystals form specific shapes"]

CODE_TASKS = [
    ("write a function that merges two sorted linked lists in place and returns the head", "linked list"),
    ("implement a thread-safe LRU cache with a fixed capacity and O(1) get/put", "cache"),
    ("parse a CSV file with quoted fields, skip blank lines, and return rows as dictionaries", "csv"),
    ("write a function that detects whether two binary trees are isomorphic", "tree"),
    ("implement an exponential backoff scheduler with jitter and a max retry cap", "scheduler"),
    ("write a rate limiter using a token bucket that supports multiple named clients", "rate limit"),
    ("implement a function that topologically sorts a directed graph and reports cycles", "graph"),
    ("write a function that splits a sentence into words keeping punctuation attached to the preceding word", "nlp"),
    ("implement a function that validates an IPv4 address including leading-zero rejection", "network"),
    ("write a function that flattens a nested list of arbitrary depth into a generator", "utils"),
    ("implement a function that finds the longest palindromic substring using dynamic programming", "string"),
    ("write a function that groups a list of events by day and sorts each group by time", "events"),
    ("implement a simple expression evaluator supporting + - * / parentheses with proper precedence", "parser"),
    ("write a function that computes the edit distance between two strings with the operation counts", "dp"),
    ("implement a function that streams a large JSON array object-by-object without loading it all", "json"),
    ("write a function that balances a binary search tree in order", "tree"),
    ("implement a function that detects a race condition-free counter with a lock and benchmark it", "concurrency"),
    ("write a function that normalizes a file path and rejects traversal attempts", "security"),
    ("implement a function that finds all anagrams in a list of words grouped by sorted letters", "string"),
    ("write a function that computes the longest common subsequence length with backtracking", "dp")]

MATHS = [
    "Solve step by step: a train leaves A at 9:00 at 60 km/h; another leaves B at 9:30 at 80 km/h toward A. The distance between A and B is 280 km. When do they meet and where?",
    "Solve step by step: the sum of three consecutive multiples of 5 is 405. What are they?",
    "Solve step by step: a rectangle's diagonal is 13 and its area is 30. What is its perimeter?",
    "Solve step by step: 15% of x plus 8 is equal to 27. Find x.",
    "Solve step by step: how many ways can you choose 3 items from 10 without order mattering?",
    "Solve step by step: a clock's hour and minute hands are 120 degrees apart between 2 and 3 o'clock. What time is it?",
    "Solve step by step: two pipes fill a tank in 4 and 6 hours; a drain empties it in 10 hours. With all three open, how long to fill the tank?",
    "Solve step by step: find the value of 2^10 - 10^2 divided by 4.",
    "Solve step by step: the ratio of boys to girls is 3:4 and there are 35 students. How many girls?",
    "Solve step by step: a number divided by 7 gives a remainder of 3; divided by 5 it gives a remainder of 2. Find the smallest such number.",
    "Solve step by step: what is 12% of 85 plus 5% of 400?",
    "Solve step by step: a ladder 5 m long leans against a wall with its base 1.4 m from the wall. How high up the wall does it reach?",
    "Solve step by step: the average of five numbers is 40. If the largest is removed the average is 32. What was the largest?",
    "Solve step by step: a circular track is 400 m. Two runners start together going opposite ways at 6 and 9 m/s. How often do they meet in one hour?",
    "Solve step by step: if a is 30% more than b, by what percent is b less than a?"]

ANALYSES = [
    "Analyze step by step: a startup's churn doubled in two months while revenue stayed flat. List the likely causes in order of impact and the metric to check for each.",
    "Analyze step by step: our batch job runs 40 minutes slower on weekends. The hardware is unchanged. What would you check first and why?",
    "Analyze step by step: two identical models differ by 3 points in accuracy. Is the difference meaningful? Describe a procedure to decide.",
    "Analyze step by step: a hospital sees a spike in flu diagnoses in January with no new virus strain. Give the most likely explanation and the data that would confirm it.",
    "Analyze step by step: a feature improved lab accuracy but decreased user retention. What does that imply and what should you do next?",
    "Analyze step by step: an A/B test shows a 12% lift with a p-value of 0.04 after a 1% sample. How would you interpret this?",
    "Analyze step by step: why does adding more data sometimes decrease a model's accuracy?",
    "Analyze step by step: a database query is 10x slower after a schema change that added an index. Explain how that could happen.",
    "Analyze step by step: a factory's defect rate drops when a new machine is installed but only on one shift. Form the hypotheses and the test for each.",
    "Analyze step by step: a recommendation system shows more clicks but less time spent. Reconcile the two signals.",
    "Analyze step by step: explain why a 99.9% uptime guarantee still allows about 52 minutes of downtime a year, and whether that number is right.",
    "Analyze step by step: a model that predicts house prices uses the year of renovation. Is that leakage, and how would you test?",
    "Analyze step by step: why can a system be individually rational for each node and collectively unstable? Give a network example.",
    "Analyze step by step: our logs show p99 latency is 10x p50 but averages are stable. What does that distribution shape suggest and what are three common causes?",
    "Analyze step by step: a company's per-unit cost rises as volume doubles. Identify the cost components that would explain this and how to verify each."]

JSON_EXTRACTS = [
    ("Extract as JSON only, no prose, the fields name, role, and years of experience from: 'Interview notes: our candidate Marta Kovac works as a lead backend engineer at a logistics company, with 9 years of professional experience after 2 more as a contractor.'",
     '{"name": "Marta Kovac", "role": "lead backend engineer", "years": 9}'),
    ("Extract as JSON only the product name, price in cents, and currency from: 'The Nova Desk Lamp, model ND-200, is listed at $49.99 USD in the spring catalog.'",
     '{"name": "Nova Desk Lamp", "price_cents": 4999, "currency": "USD"}'),
    ("Extract as JSON only the fields from: 'The incident on 2026-11-01 started at 14:23 UTC, was caused by a bad deployment, and took 3 engineers and 47 minutes to resolve.'",
     '{"date": "2026-11-01", "start": "14:23", "cause": "bad deployment", "engineers": 3, "duration_min": 47}'),
    ("Extract as JSON only the fields from: 'The flight AA1042 from JFK to LAX departs 07:45, arrives 10:50, gate B22, and is on time.'",
     '{"flight": "AA1042", "from": "JFK", "to": "LAX", "depart": "07:45", "arrive": "10:50", "gate": "B22"}'),
    ("Extract as JSON only the fields from: 'Patient record: 34-year-old, 72 kg, height 1.78 m, blood pressure 121/78, heart rate 61 bpm at rest.'",
     '{"age": 34, "weight_kg": 72, "height_m": 1.78, "bp_systolic": 121, "bp_diastolic": 78, "hr": 61}'),
    ("Extract as JSON only the fields from: 'The order #A7712 contains 2x USB-C hub at $18.50 each and 1x cable at $9.00, total $46.00, paid by card.'",
     '{"order": "A7712", "items": [{"name": "USB-C hub", "qty": 2, "price": 18.5}, {"name": "cable", "qty": 1, "price": 9.0}], "total": 46.0, "payment": "card"}'),
    ("Extract as JSON only the fields from: 'Meeting notes: the decision was to postpone the launch by two weeks, owned by Priya, with a review on Friday.'",
     '{"decision": "postpone launch", "delay_weeks": 2, "owner": "Priya", "review": "Friday"}'),
    ("Extract as JSON only the fields from: 'Sensor reading: the air quality index is 42, temperature 21.3 C, humidity 58%, recorded at 08:00 on 2026-11-02 in Berlin.'",
     '{"aqi": 42, "temp_c": 21.3, "humidity": 58, "time": "08:00", "date": "2026-11-02", "city": "Berlin"}'),
    ("Extract as JSON only the fields from: 'The ticket was opened by j.smith, labeled bug and p2, assigned to the storage team, and closed after 3 days.'",
     {"opened_by": "j.smith", "labels": ["bug", "p2"], "team": "storage", "closed_days": 3}),
    ("Extract as JSON only the fields from: 'Recipe: pasta with tomato sauce needs 300 g pasta, 400 g tomatoes, 2 garlic cloves, 20 ml olive oil, salt, and 12 minutes of cooking.'",
     '{"pasta_g": 300, "tomatoes_g": 400, "garlic_cloves": 2, "olive_oil_ml": 20, "cook_min": 12}')]


def main():
    out = []
    for i, t in enumerate(TOPICS * 2):
        for style in ("Write a 250-word essay about ",
                      "Explain like I'm 12, in one paragraph, "):
            out.append({"style": "essay", "prompt": style + t})
    for i, q in enumerate(QA_TOPICS * 3):
        out.append({"style": "qa", "prompt": f"Answer in a few sentences: {q}?"})
    for i, (task, _) in enumerate(CODE_TASKS * 3):
        out.append({"style": "code", "prompt": f"Write a clean Python function. {task}."})
    for i, m in enumerate(MATHS * 4):
        out.append({"style": "math", "prompt": m})
    for i, a in enumerate(ANALYSES * 4):
        out.append({"style": "analysis", "prompt": a})
    for i, (j, _) in enumerate(JSON_EXTRACTS * 4):
        out.append({"style": "json", "prompt": j})
    random.shuffle(out)
    p = Path(__file__).with_name("serve_styles.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} prompts -> {p}")


if __name__ == "__main__":
    main()
