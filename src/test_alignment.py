raw = "I can't remember"

mwt = "can't"
parts = ["ca", "n't"]

print("RAW :", raw)
print("MWT :", mwt)
print("PARTS:", " | ".join(parts))

position = 0

for part in parts:
    position += len(part)
    print(f"Boundary after '{part}': character position {position}")