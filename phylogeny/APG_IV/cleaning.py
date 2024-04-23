import pandas as pd

apg = pd.read_csv("APG_IV.csv")
print(apg)
# apg = apg[apg["family"].str.contains(r"\[")].reset_index(
#     drop=True
# )  # .reset_index(drop=True)
print(apg)
apg["family"] = apg["family"].str.replace(r"^.*?([A-Z].*)$", r"\1", regex=True)
apg["family"] = apg["family"].str.split(" ").str[0]
print(apg)

apg.to_csv("./APG_IV_clean.csv", index=False)

print(apg)
