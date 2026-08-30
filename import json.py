import json
import pandas as pd

with open("nvdcve-2.0-2026.json", "r") as f:
    data = json.load(f)

vulnerabilities = data["vulnerabilities"]

print(len(vulnerabilities))

rows = []

for item in vulnerabilities:

    cve = item["cve"]

    # Basic information
    cve_id = cve.get("id")
    published = cve.get("published")
    last_modified = cve.get("lastModified")

    # Description
    description = None

    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            description = d.get("value")
            break

    # CVSS information
    cvss_score = None
    severity = None
    attack_vector = None
    attack_complexity = None
    privileges_required = None
    user_interaction = None

    metrics = cve.get("metrics", {})

    # Prefer CVSS 3.1
    if "cvssMetricV31" in metrics:

        cvss = metrics["cvssMetricV31"][0]["cvssData"]

        cvss_score = cvss.get("baseScore")
        severity = cvss.get("baseSeverity")
        attack_vector = cvss.get("attackVector")
        attack_complexity = cvss.get("attackComplexity")
        privileges_required = cvss.get("privilegesRequired")
        user_interaction = cvss.get("userInteraction")

    # CWE
    cwe = None

    weaknesses = cve.get("weaknesses", [])

    if weaknesses:
        descriptions = weaknesses[0].get("description", [])

        if descriptions:
            cwe = descriptions[0].get("value")

    # Whether vulnerability is in CISA Known Exploited Vulnerabilities
    exploited = "cisaExploitAdd" in cve

    rows.append({
        "cve_id": cve_id,
        "published": published,
        "last_modified": last_modified,
        "cvss_score": cvss_score,
        "severity": severity,
        "attack_vector": attack_vector,
        "attack_complexity": attack_complexity,
        "privileges_required": privileges_required,
        "user_interaction": user_interaction,
        "cwe": cwe,
        "exploited": exploited,
        "description": description
    })

df = pd.DataFrame(rows)

df.head()

df["published"] = pd.to_datetime(df["published"])
df["last_modified"] = pd.to_datetime(df["last_modified"])
df["date"] = df["published"].dt.date

df["month"] = df["published"].dt.to_period("M").astype(str)

df["week"] = (
    df["published"]
    .dt.to_period("W")
    .apply(lambda x: x.start_time)
)

df.to_csv("nvd_cve_2026.csv", index=False)