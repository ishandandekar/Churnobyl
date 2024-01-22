import subprocess

subprocess.run(
    ["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn"],
    check=True,
)
