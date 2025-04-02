from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tabulate import tabulate
import json

class ResultsService:
    @staticmethod
    def calculate_metrics_from_results(results: List[Dict]) -> Tuple[int, float, float, float, float]:
        if not results:
            return 0, 0.0, 0.0, 0.0, 0.0

        total = len(results)
        avg_sim = np.mean([r["metrics"]["avg_similarity_score"] for r in results])
        max_sim = np.mean([r["metrics"]["max_similarity_score"] for r in results])
        min_sim = np.mean([r["metrics"]["min_similarity_score"] for r in results])
        std_sim = np.mean([r["metrics"]["similarity_std"] for r in results])

        return total, avg_sim, max_sim, min_sim, std_sim

    @staticmethod
    def save_results(
            all_results: List[Dict],
            table_data: List[List],
            output_json: Path,
            output_csv: Path,
            timestamp: str
    ) -> None:
        with open(output_json, 'w', encoding='utf-8') as f:
            json_out = {
                "timestamp": timestamp,
                "databases": all_results
            }
            json.dump(json_out, f, ensure_ascii=False, indent=2)

        headers = ["Database", "Questions Processed", "Average Similarity", "Max Similarity", "Min Similarity", 
                  "Standard Deviation"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        df = pd.DataFrame(table_data, columns=headers)
        df.to_csv(output_csv, index=False)

        print("\nVector Database Evaluation Results:")
        print(table)
        print(f"\nDetailed results saved to {output_json}")
        print(f"Results table saved to {output_csv}") 