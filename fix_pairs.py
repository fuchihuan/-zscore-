import pathlib

f = pathlib.Path(r'c:\Users\buvh8\OneDrive\桌面\資料夾\pair_trading\batch_backtest.py')
content = f.read_text(encoding='utf-8')

# Replace pair 2 ticker symbols
content = content.replace("'sym1': '3264.TWO',  'sym2': '2368.TW'", "'sym1': '8039.TW',   'sym2': '2327.TW'")
# Replace pair 2 names and industry
content = content.replace("'name1': '\u6b23\u92d3',   'name2': '\u91d1\u50cf\u96fb',  'industry': 'IC\u6e2c\u8a66/PCB'", "'name1': '\u53f0\u8679',   'name2': '\u570b\u5de8',    'industry': '\u96fb\u5b50\u6750\u6599/\u88ab\u52d5\u5143\u4ef6'")
# Fix typo in pair 9
content = content.replace("'name1': '\u7daf\u7a4e',", "'name1': '\u7daf\u7a4e',")

f.write_text(content, encoding='utf-8')
print("Done - pairs updated")

# Verify
for line in content.split('\n'):
    if "'id': 2," in line or "'id': 8," in line or "'id': 9," in line:
        print(line.strip())
