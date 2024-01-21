with open("all_embeddings.csv", "r") as file:
    lines = file.readlines()[:100]

with open("subset_embeddings.csv", "w") as file:
    file.writelines(lines)
