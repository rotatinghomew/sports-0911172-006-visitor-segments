import random
import json
from datetime import date, timedelta

def generate():
    rng = random.Random(117206)
    operations = []

    def add(command, **arguments):
        operations.append([command, arguments])

    def day(offset):
        return (date(2026, 7, 1) + timedelta(days=offset)).isoformat()
    for index in range(120):
        add('identity.connect', id=f'p{index}', devices=[f'd{index}', f'family{index // 3}'], consent=index % 9 != 0)
    for index in range(840):
        add('visits.record', id=f'v{index}', device=f'd{index % 120}' if index % 4 else f'family{index % 40}', venue=f'v{index % 4}', at=day(index // 12), activity=rng.choice(['match', 'exercise', 'mall', 'other']), cents=rng.choice([0, 0, 4000, 8000]))
    for index in range(0, 120, 13):
        add('identity.split', id=f'p{index}', devices=[f'd{index}'])
    for index in range(0, 120, 17):
        add('identity.withdraw', id=f'p{index}')
    return operations

def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成本地合成业务记录，不包含真实个人资料')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from pathlib import Path
    Path(args.output).write_text(json.dumps(generate(), ensure_ascii=False, indent=2) + '\n')

if __name__=='__main__':
    main()
