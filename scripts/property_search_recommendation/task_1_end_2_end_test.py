import asyncio
import json
import os
from glob import glob
from typing import Dict

from infrastructures.neo4j.client import neo4j_client
from infrastructures.neo4j.retriever import hybrid_search
from services.property_search_recommendation.models import RealEstateQuery
from services.property_search_recommendation.service import extract_user_question_intent

DATASET_DIR = "../../data/testing_dataset_twhg_with_latlng_and_places"
REPORT_FILE = "../../reports/task_1/evaluation_report.md"


async def process_question(question_data: Dict, target_property_id: str) -> bool:
    question = question_data["question"]
    try:
        user_intent: RealEstateQuery = await extract_user_question_intent(user_query=question)
        results = await hybrid_search(query=user_intent, graph_limit=200, topk=10)

        found = False
        for r in results:
            if r['property_id'] == target_property_id:
                found = True
                break
        return found
    except Exception as e:
        print(f"Error processing question: {question}. Error: {e}")
        return False


async def process_property(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        property_id = data.get("property_id")
        question_list = data.get("question_list", [])

        print(f"Processing property: {property_id} from {os.path.basename(file_path)}")

        success_count = 0
        results = []

        question_tasks = [process_question(q, property_id) for q in question_list]
        question_outcomes = await asyncio.gather(*question_tasks)

        for q, is_found in zip(question_list, question_outcomes):
            if is_found:
                success_count += 1
            results.append({
                "question": q["question"],
                "found": is_found
            })

        return {
            "property_id": property_id,
            "file_name": os.path.basename(file_path),
            "total_questions": len(question_list),
            "success_count": success_count,
            "recall_rate": success_count / len(question_list) if question_list else 0,
            "details": results
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {
            "property_id": "ERROR",
            "file_name": os.path.basename(file_path),
            "total_questions": 0,
            "success_count": 0,
            "recall_rate": 0,
            "details": []
        }


async def main():
    files = sorted(glob(os.path.join(DATASET_DIR, "*.json")))
    print(f"Found {len(files)} files in {DATASET_DIR}")

    report_lines = [
        "# End-to-End Retrieval Evaluation Report",
        "",
        "| Property ID | File Name | Recall Rate | Success/Total |",
        "|---|---|---|---|"
    ]

    # Batch processing: 2 files at a time
    batch_size = 2
    all_results = []

    try:
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            print(
                f"Processing batch {i // batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}: {[os.path.basename(f) for f in batch_files]}")

            tasks = [process_property(f) for f in batch_files]
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)

            for res in batch_results:
                line = f"| {res['property_id']} | {res['file_name']} | {res['recall_rate']:.2f} | {res['success_count']}/{res['total_questions']} |"
                report_lines.append(line)

            # Write report incrementally
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))

    finally:
        await neo4j_client.close()

    print(f"Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
