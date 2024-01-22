from pathlib import Path

churnobyl = Path("./churnobyl")
for file in churnobyl.glob("*.py"):
    print("#" * 6, end="\n\n")
    print(file.name)
    print(type(file))
