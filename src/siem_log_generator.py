# src/siem_log_generator.py

"""
Generate SIEM/ELK-style logs from processed data
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import random


def generate_elk_logs():
    """Convert processed data to ELK/Splunk log format"""

    print("Generating SIEM/ELK style logs...")

    # Load processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    train_path = os.path.join(project_root, 'data', 'processed', 'train_data.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'test_data.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine data
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Generate SIEM-like log entries
    siem_logs = []

    # Sample log templates (based on common SIEM formats)
    log_templates = [
        {
            'timestamp': None,
            'source': 'firewall',
            'event_type': 'network_connection',
            'src_ip': '10.0.0.{}',
            'dst_ip': '192.168.1.{}',
            'src_port': None,
            'dst_port': 443,
            'protocol': 'TCP',
            'bytes_sent': None,
            'bytes_received': None,
            'duration': None,
            'action': 'ALLOW',
            'severity': 'INFO'
        },
        {
            'timestamp': None,
            'source': 'ids',
            'event_type': 'bruteforce_attempt',
            'src_ip': '10.0.0.{}',
            'dst_ip': '192.168.1.{}',
            'src_port': None,
            'dst_port': 443,
            'protocol': 'TCP',
            'bytes_sent': None,
            'bytes_received': None,
            'duration': None,
            'action': 'ALERT',
            'severity': 'HIGH'
        },
        {
            'timestamp': None,
            'source': 'web_server',
            'event_type': 'http_request',
            'src_ip': '10.0.0.{}',
            'dst_ip': '192.168.1.{}',
            'src_port': None,
            'dst_port': 443,
            'protocol': 'HTTPS',
            'bytes_sent': None,
            'bytes_received': None,
            'method': 'POST',
            'url': '/wp-login.php',
            'user_agent': 'Mozilla/5.0',
            'response_code': 200
        }
    ]

    # Generate logs
    base_time = datetime.now() - timedelta(days=7)

    for i, (_, row) in enumerate(df.iterrows()):
        # Select template based on class
        if row['CLASS'] == 1:  # Attack
            template_idx = 1  # IDS alert template
        else:
            template_idx = random.choice([0, 2])  # Normal traffic

        template = log_templates[template_idx].copy()

        # Fill template with data
        template['timestamp'] = (base_time + timedelta(seconds=i * 10)).isoformat()
        template['src_ip'] = template['src_ip'].format(random.randint(1, 254))
        template['dst_ip'] = template['dst_ip'].format(random.randint(1, 254))

        # Map features to log fields
        if 'feature_0' in df.columns:
            template['src_port'] = int(10000 + (row['feature_0'] * 1000))
            template['bytes_sent'] = int(abs(row.get('feature_3', 0) * 1000))
            template['bytes_received'] = int(abs(row.get('feature_4', 0) * 1000))
            template['duration'] = float(row.get('feature_2', 0))

        # Add ML prediction
        template['ml_prediction'] = int(row['CLASS'])
        template['ml_confidence'] = float(random.uniform(0.85, 0.99))

        siem_logs.append(template)

        if i % 10000 == 0:
            print(f"Generated {i} logs...")

    # Save as JSON (ELK format)
    output_dir = os.path.join(project_root, 'data', 'siem_logs')
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON lines (for ELK)
    jsonl_path = os.path.join(output_dir, 'bruteforce_logs.jsonl')
    with open(jsonl_path, 'w') as f:
        for log in siem_logs:
            f.write(json.dumps(log) + '\n')

    # Save as CSV (for Splunk)
    csv_path = os.path.join(output_dir, 'bruteforce_logs.csv')
    siem_df = pd.DataFrame(siem_logs)
    siem_df.to_csv(csv_path, index=False)

    print(f"\nGenerated {len(siem_logs)} SIEM/ELK logs")
    print(f"JSONL (ELK format): {jsonl_path}")
    print(f"CSV (Splunk format): {csv_path}")

    return siem_logs


def create_elk_index_template():
    """Create ELK index template for the logs"""

    template = {
        "index_patterns": ["bruteforce-logs-*"],
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "source": {"type": "keyword"},
                    "event_type": {"type": "keyword"},
                    "src_ip": {"type": "ip"},
                    "dst_ip": {"type": "ip"},
                    "src_port": {"type": "integer"},
                    "dst_port": {"type": "integer"},
                    "protocol": {"type": "keyword"},
                    "bytes_sent": {"type": "integer"},
                    "bytes_received": {"type": "integer"},
                    "duration": {"type": "float"},
                    "action": {"type": "keyword"},
                    "severity": {"type": "keyword"},
                    "ml_prediction": {"type": "integer"},
                    "ml_confidence": {"type": "float"}
                }
            }
        }
    }

    output_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(output_dir, '..', 'config', 'elk_index_template.json')

    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"ELK index template saved to: {template_path}")
    return template


def create_splunk_config():
    """Create Splunk configuration for data ingestion"""

    config = {
        "splunk_inputs": {
            "monitor": [
                {
                    "path": "/path/to/bruteforce_logs.jsonl",
                    "sourcetype": "bruteforce:logs",
                    "index": "bruteforce_security"
                }
            ]
        },
        "splunk_fields": [
            {"field": "src_ip", "type": "ip"},
            {"field": "dst_ip", "type": "ip"},
            {"field": "ml_prediction", "type": "number"},
            {"field": "severity", "type": "string"}
        ]
    }

    output_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(output_dir, '..', 'config', 'splunk_config.json')

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Splunk configuration saved to: {config_path}")
    return config


def main():
    """Main function to generate SIEM logs"""
    print("=" * 60)
    print("SIEM/ELK LOG GENERATION")
    print("=" * 60)

    # Generate logs
    logs = generate_elk_logs()

    # Create configurations
    elk_template = create_elk_index_template()
    splunk_config = create_splunk_config()

    print("\n" + "=" * 60)
    print("SIEM LOG GENERATION COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")
    print("- SIEM logs: data/siem_logs/")
    print("- ELK template: config/elk_index_template.json")
    print("- Splunk config: config/splunk_config.json")

    return logs


if __name__ == "__main__":
    main()