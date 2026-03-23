import json
import os
from convert_to_json import clean_job_data

def run_test():
    test_file = "test_data.json"
    test_data = [
        {"id": 1, "job_description": "title: junior als software engineer"},
        {"id": 2, "job_description": "Senior project manager"},
        {"id": 3, "job_description": "Werkstudent:in als Data Scientist"},
        {"id": 4, "job_description": "I can't provide this information."},
        {"id": 5, "job_description": "I'm sorry, I'm just an AI."},
        {"id": 6, "job_description": "regular job description"},
        {"id": 7, "job_description": "title: software engineer"},.
        {"id": 8, "job_description": "internship developer"},
        {"id": 9, "job_description": "Titel: Praktikant als Analyst"},
        {"id": 10, "job_description": "praktikum im Bereich Marketing"}
    ]
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("Running cleaning function...")
    clean_job_data(test_file)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    print("\nResults:")
    for entry in processed_data:
        print(f"ID {entry['id']}: '{entry['job_description']}'")
    
    # Check deletions
    ids = [e['id'] for e in processed_data]
    if 4 not in ids and 5 not in ids:
        print("\nSUCCESS: Failure patterns removed.")
    else:
        print("\nFAILURE: Failure patterns NOT removed.")

    # Specific checks
    expected = {
        1: "software engineer",
        2: "project manager",
        3: "Data Scientist",
        6: "regular job description",
        7: "title: software engineer",
        8: "developer",
        9: "Analyst",
        10: "im Bereich Marketing"
    }
    
    all_passed = True
    for eid, text in expected.items():
        # find entry
        entry = next((e for e in processed_data if e['id'] == eid), None)
        if entry:
            if entry['job_description'] == text:
                print(f"ID {eid}: PASS")
            else:
                print(f"ID {eid}: FAIL (Expected '{text}', got '{entry['job_description']}')")
                all_passed = False
        elif eid in [4, 5]:
            continue
        else:
            print(f"ID {eid}: MISSING")
            all_passed = False
            
    if all_passed:
        print("\nOVERALL: ALL TESTS PASSED")
    else:
        print("\nOVERALL: SOME TESTS FAILED")

    os.remove(test_file)

if __name__ == "__main__":
    run_test()
