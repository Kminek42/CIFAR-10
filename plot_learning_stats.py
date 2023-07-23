import matplotlib.pyplot as plt

stats_file = open(file="stats.csv", mode="r")
stats_data = stats_file.read().split(sep="\n")
stats_file.close()

stats_data = [[float(num) for num in row.split(sep=",")] for row in stats_data if len(row)]

loss = [row[0] for row in stats_data]
acc = [row[1] for row in stats_data]

plt.plot(loss)
plt.show()

plt.plot(acc)
plt.show()
