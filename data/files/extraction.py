import json

with open("data\jsons\quanguo1_tmp.json", 'r', encoding="utf-8") as f:
    data = json.loads(f.read())

output = []
for d in data:
    output.append(d["题号"] + ". " + d["题目"])

with open("data\jsons\quanguo1.json", 'w', encoding="utf-8") as f:
    f.write(json.dumps(output, indent=4, ensure_ascii=False))