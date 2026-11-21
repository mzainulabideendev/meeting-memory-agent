import sys
sys.path.insert(0,'src')
from meeting_agent.config import get_settings
from meeting_agent.retrieval.query_analyzer import analyze_question, plan_queries
from meeting_agent.agents.researcher import Researcher

settings = get_settings()
researcher = Researcher(settings)

def run(qid, question, team):
    analysis = analyze_question(question, team_hint=team)
    plan = plan_queries(analysis)
    bundle = researcher.gather(plan)
    print('====', qid, '| intents:', analysis.intents, '| requires_numbers:', analysis.requires_numbers, '| mentions_change:', analysis.mentions_change)
    print('  expansion_terms:', analysis.expansion_terms[:14])
    print('  bundle meetings:', bundle.meetings, 'chunks:', len(bundle.chunks), 'sufficient:', bundle.sufficient)
    for c in bundle.chunks[:8]:
        print('   [', c.meeting_id, round(c.score,2), ']', c.text[:150].replace('\n',' '))
    print()

run('E02','What power source did TS3012 settle on, and did they adopt the rechargeable option that was discussed?','TS3012')
run('E03',"How did TS3012's position on the LCD screen change between meetings b and d?",'TS3012')
run('E05','What did TS3012 know about finances in meeting c, and what did the final costing in meeting d show?','TS3012')
