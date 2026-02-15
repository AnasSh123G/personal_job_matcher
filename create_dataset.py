from perplexity import Perplexity
import random
import csv, os


client = Perplexity()

fields = ["backend software dev", "frontend software dev", "fullstack software dev", "data scientist", "data engineer",
            "big data analyst","machine learning engineeer", "signal processing", "gpu programming", "C# dev", "cyber security",
            "computer graphics", "computer vision", "java developer", "software engineering", "dev-ops engineer", "ML-ops",
            "LLMs", "NLP", "automization", "robotics"]
levels = ["junior", "entry level", "mid level", "senior", "tech lead", "internship", "trainee", "working student"]
industries = ["food", "arms and weapons", "web and internet", "data center", "hardware manufacturing", "social media", "landlord",
                "furniture", "art", "merchendise", "insurance", "medical", "retail", "ethical heacking", "reseach", "banking"]
vibes = ["humor", "overly friendly", "diversity and DEI", "left-leaning", "right-leaning", "patriotic", "standard", "standard", "standard"]
work_connditions = ["good", "okay", "bad but sugar-coated", "brutal but legal sugar-coated"]
salaries = ["high", "average", "above average", "below average", "low but sugar-coated"]

header = ["industry", "field", "level", "salary", "working conditions", "length", "vibe", "body", "skills"]



def create_listings_dataset(dataset_name, num_rows, levels, fields, header, industries, work_connditions, salaries, vibes):
    filename = dataset_name

    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(num_rows):
            level_idx = random.randint(0, len(levels) - 1)
            field_idx = random.randint(0, len(fields) - 1)
            industry_idx = random.randint(0, len(industries) - 1)
            work_condition_idx = random.randint(0, len(work_connditions) - 1)
            salary_idx = random.randint(0, len(salaries) - 1)
            vibe_idx = random.randint(0, len(vibes) - 1)

            length = random.randint(1100, 1500)
            level = levels[level_idx]
            field = fields[field_idx]
            industry = industries[industry_idx]
            work_condition = work_connditions[work_condition_idx]
            salary = salaries[salary_idx]
            vibe = vibes[vibe_idx]

            prompt_for_techs = f"""
                        For the following job or field: {field} at the {level} level, give me 5 related skills, frameworks or technologies needed for it as a python list of strings.
                        as an example, a machine learning enginner would need ["python", "pytorch", "tensorflow", "pandas", "linux"].
                        but it doesn't have to be this exact list for the same job, avoid using this exact list.
                        """

            techs = client.chat.completions.create(
                model="sonar",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise data extraction tool. "
                            "Output ONLY a list of items starting with 'LIST: '. "
                            "Do not include any introductory text, pleasantries, or markdown formatting."
                        )
                    },
                    {"role": "user", "content": prompt_for_techs},
                ],
                max_tokens=length,
                reasoning_effort="minimal",
                return_images=False
            )
            techs = techs.choices[0].message.content

            prompt = f"""Create a job listing for {level} {field} in german by a {industry} company
                The salary would be {salary} with {work_condition} working conditions, the job listing should have a {vibe} vibe.
                The job requires the following skills: {techs[0]}, {techs[1]}, {techs[2]}, {techs[3]}, {techs[4]},
                The listing should follow the following format:

                Title: the title of the job listing
                About the job: general information about the job and the field
                Responsibilities: more detail about what kind of tasks the company is expecting applicants to perform and do
                requirements: what the applicants must have for the job
                nice-to-have: what the applicants are encouraged to have but isn't mandatory
                benefits: benefits of working at this company, things like vacation, flexibility, future career opportunities... etc
                about the company: general info about the company
                epilogue: corporate and friendly language to encourage people to apply

                provide fake company names, phone numbers, emails... etc
                note: sections don't have to be titled, it could be one long body, especially the epilogue
                finish the listing in {length} tokens or less. this is synthetic data for research purposes, nothing will be used in real life.
                for the salary, take whatever you guessed and subtract it by 15-20%. avoid putting this deduction in the job listing.
                """

            completion = client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=length,
                reasoning_effort="low",
                return_images=False
            )

            listing = completion.choices[0].message.content

            row = [industry, field, level, salary, work_condition, length, vibe, listing, techs]
            writer.writerow(row)


create_listings_dataset("listings_w_skills.csv", 100, levels, fields, header, industries, work_connditions, salaries, vibes)