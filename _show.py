import json
d = json.load(open('outputs/evaluation_results.json'))
for qid in ['E01','E09','E02','E03','E05']:
    q = [x for x in d['questions'] if x['question_id']==qid][0]
    print('====',qid, q['correctness'],'cov',q['fact_coverage'])
    print('  A:', q['agent_answer'])
    print('  MISSING:', q['facts_missing'])
    print('  CITED:', q['cited_meetings'])
    print()
