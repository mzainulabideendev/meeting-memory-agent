import re
files = {
 'TS3012b': r'D:\meeting memory agent\data\Catalist_AI_Intern_Build_Test_Data\transcripts\TS3012b.txt',
 'TS3012c': r'D:\meeting memory agent\data\Catalist_AI_Intern_Build_Test_Data\transcripts\TS3012c.txt',
 'TS3012d': r'D:\meeting memory agent\data\Catalist_AI_Intern_Build_Test_Data\transcripts\TS3012d.txt',
}
texts = {k: open(v, encoding='utf-8', errors='replace').read() for k,v in files.items()}

patterns = {
 'regular batteries': r'regular batteries',
 'rechargeable': r'rechargeable',
 'lose the LCD': r'lose the L',
 'consideration LCD': r'consideration.*L_C_D',
 'ninety percent': r'ninety percent',
 'finances': r'finances',
 'seventeen': r'seventeen',
 'twelve and a half': r'twelve and a half',
 'twelve fifty': r'twelve fifty',
 'beep': r'beep',
 'flash': r'flash',
 'twenty five': r'twenty five',
 'speech recognition': r'speech recognition',
 'rubber': r'rubber',
}
for name, pat in patterns.items():
    print('=== %s ===' % name)
    for fid, text in texts.items():
        for m in re.finditer(pat, text, re.I):
            s = max(0, m.start()-120); e = min(len(text), m.end()+120)
            snippet = text[s:e].replace('\n',' | ')
            print('  %s: ...%s...' % (fid, snippet))
            break
