# analysis_attacks.py

import pandas as pd
import numpy as np


def analyze_attack_types():
    # Загружаем данные
    df = pd.read_csv('data/raw/https_bruteforce_samples.csv')

    print("=" * 60)
    print("DETAILED ATTACK ANALYSIS")
    print("=" * 60)

    # 1. Все атаки в датасете
    attack_df = df[df['CLASS'] == 1]

    print(f"\nTotal attacks in dataset: {len(attack_df):,}")
    print(f"Total benign traffic: {(df['CLASS'] == 0).sum():,}")

    # 2. Распределение по инструментам
    print("\n" + "=" * 60)
    print("ATTACKS BY TOOL:")
    print("=" * 60)

    # Извлекаем инструмент из SCENARIO
    def extract_tool(scenario):
        if 'hydra' in scenario.lower():
            return 'Hydra'
        elif 'ncrack' in scenario.lower():
            return 'Ncrack'
        elif 'patator' in scenario.lower():
            return 'Patator'
        else:
            return 'Unknown'

    attack_df['tool'] = attack_df['SCENARIO'].apply(extract_tool)
    tool_dist = attack_df['tool'].value_counts()

    for tool, count in tool_dist.items():
        percentage = (count / len(attack_df)) * 100
        print(f"{tool}: {count:,} attacks ({percentage:.1f}%)")

    # 3. Распределение по целевым приложениям
    print("\n" + "=" * 60)
    print("ATTACKS BY TARGET APPLICATION:")
    print("=" * 60)

    def extract_app(scenario):
        # Извлекаем название приложения из SCENARIO
        apps = ['wordpress', 'joomla', 'mediawiki', 'ghost', 'grafana',
                'discourse', 'phpbb', 'opencart', 'redmine', 'nginx', 'apache']

        for app in apps:
            if app in scenario.lower():
                return app.capitalize()
        return 'Unknown'

    attack_df['application'] = attack_df['SCENARIO'].apply(extract_app)
    app_dist = attack_df['application'].value_counts().head(10)

    for app, count in app_dist.items():
        percentage = (count / len(attack_df)) * 100
        print(f"{app}: {count:,} attacks ({percentage:.1f}%)")

    # 4. Статистики по атакам
    print("\n" + "=" * 60)
    print("ATTACK STATISTICS:")
    print("=" * 60)

    # Средние значения для атак
    attack_stats = attack_df.select_dtypes(include=[np.number]).mean()
    benign_stats = df[df['CLASS'] == 0].select_dtypes(include=[np.number]).mean()

    # Сравниваем ключевые метрики
    key_metrics = ['DURATION', 'BYTES', 'PACKETS', 'roundtrips', 'bytes_per_sec']

    print("\nAverage values comparison:")
    print(f"{'Metric':<20} {'Attacks':<15} {'Benign':<15} {'Ratio'}")
    print("-" * 60)

    for metric in key_metrics:
        if metric in attack_stats.index:
            attack_val = attack_stats[metric]
            benign_val = benign_stats[metric]
            ratio = attack_val / benign_val if benign_val > 0 else float('inf')

            print(f"{metric:<20} {attack_val:<15.1f} {benign_val:<15.1f} {ratio:.1f}x")


if __name__ == "__main__":
    analyze_attack_types()