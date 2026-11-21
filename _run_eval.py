import sys, asyncio
sys.path.insert(0, 'src')
from meeting_agent.evaluation.evaluator import run_evaluation
res = asyncio.run(run_evaluation(regenerate=True))
s = res['summary']
print('=== BASELINE SUMMARY ===')
for k,v in s.items():
    print(k, ':', v)
print('=== PER QUESTION ===')
for e in res['evaluations']:
    print(e.question_id, '|', e.correctness, '| cov', e.fact_coverage, '| state', e.decision_state_accuracy, '| cit', e.citation_validity, '| retr', e.meeting_retrieval_recall)
