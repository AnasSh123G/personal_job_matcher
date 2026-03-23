import json
import re
import os
import csv
import ast

def convert_csv_to_json(input_csv, output_json):
    known_skills_raw = ["SQL", "LLMs", "LLM", "Pytorch", "Azure", "Python", "OOP", "git", "Linux", "Docker", "Power BI",
                        "PowerBI", "OpenCV", "SciPy", "NumPy", "postgresql", "huggingface", "hugging face", "pandas",
                        "transformers", "scikit-learn", "scikit learn", "django", "NestJS", "Typescript"]
    known_skills = {s.strip().lower() for s in known_skills_raw}

    def normalize_skill(s: str) -> str:
        return s.strip().lower()

    items = []

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            raw = row["skills"]

            match = re.search(r'\[[^\]]*\]', raw)
            if not match:
                raise ValueError("No valid list found")

            cleaned = match.group(0)

            target_skills_raw = ast.literal_eval(cleaned)
            target_skills_list = [normalize_skill(skill) for skill in target_skills_raw]

            missing_skills_list = []
            for skill in target_skills_list:
                if skill not in known_skills:
                    missing_skills_list.append(skill)

            item = {
                "id": idx,
                "job_title":"",
                "job_description": row["body"],
                "target_skills": target_skills_list,
                "career_level": row["level"],
                "missing_skills": missing_skills_list,
                "recommended": "",
                "match_score": ""
            }
            items.append(item)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def clean_job_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {file_path}.")
            return

    if not isinstance(data, list):
        print("Error: JSON data is not a list.")
        return

    prefixes = [
        "tech lead", "junior", "senior", "praktikum", "praktikant", 
        "internship", "ausbilder", "working student", "werkstudent", 
        "trainee", "Einstiegsstelle:", "Stellenanzeige", "Werkstudent:in"
    ]
    
    prefixes.sort(key=len, reverse=True)
    
    escaped_prefixes = [re.escape(p) for p in prefixes]
    prefix_pattern = "|".join(escaped_prefixes)
    
    cleaning_regex = re.compile(
        rf"^([\s\*\#\-]*)(?:title:|titel:)?\s*(?:{prefix_pattern})(?:\s+als)?\s*", 
        re.IGNORECASE
    )

    deletion_patterns = ["I can't", "I'm sorry", "I appreciate"]
    
    new_data = []
    for entry in data:
        job_desc = entry.get("job_description", "")
        
        check_text = re.sub(r'^[\s\*\#\-]+', '', job_desc).strip().lower()
        
        should_delete = False
        for pattern in deletion_patterns:
            if check_text.startswith(pattern.lower()):
                should_delete = True
                break
        
        if should_delete:
            continue
            
        cleaned_desc = cleaning_regex.sub(r"\1", job_desc)
        entry["job_description"] = cleaned_desc
        
        nl_idx = cleaned_desc.find("\n")
        slash_d_idx = cleaned_desc.lower().find("/d)")
        
        nl_idx = cleaned_desc.find("\n")
        slash_d_idx = cleaned_desc.lower().find("/d)")
        
        end_indices = []
        if nl_idx != -1:
            end_indices.append(nl_idx)
        if slash_d_idx != -1:
            end_indices.append(slash_d_idx + 3)
            
        if end_indices:
            end_idx = int(min(end_indices))
            potential_title = cleaned_desc[:end_idx].strip()
        else:
            potential_title = cleaned_desc.strip()
            
        clean_potential_title = re.sub(r'^[\s\*\#\-]+', '', potential_title).strip()
        
        if len(potential_title) >= 9:
            if len(potential_title) > 100:
                print(f"Entry ID {entry.get('id', 'Unknown')}: Title too long ({len(potential_title)} chars)")
            else:
                entry["job_title"] = clean_potential_title
                
        new_data.append(entry)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(data)} entries. {len(new_data)} entries remain.")


def scoring(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {file_path}.")
            return

    if not isinstance(data, list):
        print("Error: JSON data is not a list.")
        return
        
    multipliers = {
        "junior": 10,
        "entry level": 10,
        "mid level": 25,
        "senior": 35,
        "tech lead": 35,
        "internship": 8,
        "trainee": 8,
        "working student": 8
    }

    for entry in data:
        missing_skills = entry.get("missing_skills", [])
        num_missing = len(missing_skills)
        
        career_level = entry.get("career_level", "").lower()
        multiplier = multipliers.get(career_level, 10)
        
        score = 100 - (num_missing * multiplier)
        if score <= 0:
            score = 0
            
        entry["match_score"] = score
        entry["recommended"] = score > 70
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Scored {len(data)} entries.")
