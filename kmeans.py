#K-Means clustering implementation

# imports
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

data_list = []

# ====
# Computes the distance between two data points
def euclidean_distance(point1, point2):
	output = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
	return output

# ====
# Reads data in from the csv files
# Returns a numpy array of the data(excluding the headings)
def read_data(path):
	with open (path, 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		return np.array(list(csv_reader)[1:])

# ====
# Finds the closest centroid to each point out of all the centroids
# Takes in a point and list of centroids
# Gets the distances between that point and every centroid, and stores them in a list
# Returns the index of the smallest distance in the list
def find_closest_centroid(point, centroids):
	distances = []
	for centroid in centroids:
		distances.append(euclidean_distance(point, centroid))
	distances = np.array(distances)
	closest_centroid = np.argmin(distances)
	return closest_centroid

# ====
# Creates a specified amount of clusters
# Chooses random points in the specified dataset
# Returns a list of the clusters
def create_clusters(dataset, no_of_clusters):
	clusters = []
	for cluster in range(no_of_clusters):
		cluster_x = random.sample(list(dataset[:,1]), k=1)
		cluster_y = random.sample(list(dataset[:,2]), k=1)
		cluster = [cluster_x, cluster_y]
		clusters.append(cluster)
	clusters = np.array(clusters, dtype=np.float)
	return clusters

# ====
# K-means algortithm implementation
def calculate_cluster_mean(data_set, clusters):
  global data_list
  new_clusters = []
  
  data_x = np.array(data_set[:, 1], dtype=np.float)
  data_y = np.array(data_set[:, 2], dtype=np.float)

  # Loop through each point & find the closest centroid to the point
  data_list = []
  points_list = []
  # Initialize convergence value
  convergence_value = 0
  for i in range(len(data_x)):
    x = data_x[i]
    y = data_y[i]
    closest_centroid = find_closest_centroid((x, y), clusters)

    # Add the point and it's closest centroid to the points and data list(data list is for use outside this function) 
    points_list.append((x, y, closest_centroid))
    data_list.append((data_set[:,0][i],x, y, closest_centroid))
    # Convergence value = summing all squared distances between each point and its closest centroid
    convergence_value += float((euclidean_distance((x,y),clusters[closest_centroid]))**2)
  print(f"Convergence Value :{round(convergence_value, 2)}")
  
  # Loop through each centroid(cluster)
    # Initialize the mean value and count.
  for centroid in range(len(clusters)):
    centroid_mean_x = 0
    centroid_mean_y = 0
    centroid_mean_count = 0
    # Loop through each point in points list and
      # Check if its closest centroid(cluster) is the centroid(cluster) we are currently checking
      # If so, add the point to the centroid's mean value, increment mean count
    for point in points_list:
      if point[2] == centroid:
        centroid_mean_count += 1
        centroid_mean_x += point[0] 
        centroid_mean_y += point[1] 
    # prevents zero division error
    if centroid_mean_count == 0:
      continue
    # Calculate the mean for the cluster
    cluster_mean_x, cluster_mean_y = centroid_mean_x / centroid_mean_count, centroid_mean_y / centroid_mean_count
    cluster_mean = [cluster_mean_x, cluster_mean_y]
    # Append each cluster mean to the list
    new_clusters.append(cluster_mean)
  # Return Cluster Mean list to be reused as new clusters
  return new_clusters

# ====
# Get user input for their dataset choice. 2008 is chosen if input invalid
def get_data_choice():
  data_choice = 1
  try:
    data_choice = int(input("Choose Dataset (Default: 2008) \n1. 2008 \n2. 1953 \n3. both \n> "))
  except:
    print("Invalid input. 2008 chosen on default")
  return data_choice

# ====
# Gets user input for number of clusters. Default = 2
def get_num_of_clusters():
  num_of_clusters = 2
  try:
    num_of_clusters = int(input("Enter number of data clusters \n> "))
  except:
    print("Invalid input. 2 chosen on default")
  if num_of_clusters < 1:
    print("Can't be less than 1 cluster. Defaulted to 2")
    num_of_clusters = 2
  return num_of_clusters

# ====
# Gets user input for number of iterations. Default = 6
def get_user_iterations():
  iterations = 6
  try:
    iterations = int(input("Enter number of algorithm iterations \n> "))
  except:
    print("Invalid input. 6 chosen on default")
  if iterations < 1:
    print("Can't be less than 1 iteration. Defaulted to 6")
    iterations = 6
  return iterations

# ====
# Initialisation procedure 

# Get and display user dataset choice
data_choice = get_data_choice()
if data_choice == 2:
  plt.title("1953")
  data_set = read_data('data1953.csv')
elif data_choice == 3:
  plt.title("2008 & 1953")
  data_set = read_data('dataBoth.csv')
else:
  plt.title("2008")
  data_set = read_data('data2008.csv')

# Set up plot lables
plt.style.use('bmh')
plt.xlabel("Birth Rate (per 1000)")
plt.ylabel("Life Expectancy (years)")

# Get number of clusters, create clusters
no_of_clusters = get_num_of_clusters()
clusters = create_clusters(data_set, no_of_clusters)

# Get number of iterations, run algorithm (iterations) times
iterations = get_user_iterations()
for i in range(iterations):
	clusters = calculate_cluster_mean(data_set, clusters)

# ====
# Coloring each cluster a different cluster and adding cluster to a data clusters list.
# Loop through all data points in data list,
# Add each point to its corresponding cluster.
# Count how many countries are in that cluster and add that onto a list
# Scatter the cluster on the plot so it has its own color(convert data cluster list into np.array, excluding the country names)
Num_of_countries_in_cluster_list = []
list_of_all_data_clusters = []
for i in range(no_of_clusters):
  data_cluster = []
  Num_of_countries = 0
  for data_point in data_list:
    if data_point[3] == i:
      data_cluster.append(data_point)
      Num_of_countries += 1
  Num_of_countries_in_cluster_list.append(Num_of_countries)      
  list_of_all_data_clusters.append(data_cluster)
  data_cluster = np.array(data_cluster)
  data_cluster = np.array(data_cluster[:,1:], dtype=np.float)
  plt.scatter(data_cluster[:,0], data_cluster[:,1], label='Cluster {}'.format(i+1))



# ====
# Print out the results for questions
#1.) The number of countries belonging to each cluster
print("\nQuestion 1:\n=======")
for i in range(no_of_clusters):
  print(f"Cluster {i+1}: {Num_of_countries_in_cluster_list[i]} countries")

#2.) The list of countries belonging to each cluster
print("\nQuestion 2:\n=======")
for i in range(no_of_clusters):
  print(f"Cluster {i+1} country list:")
  for country in list_of_all_data_clusters[i]:
    print(f"\t{country[0]}")


#3.) The mean Life Expectancy and Birth Rate for each cluster
print("\nQuestion 3:\n=======")
for i in range(no_of_clusters):
  mean_birth_rate = 0
  mean_life_expectancy = 0
  count = 0
  print(f"Cluster {i+1} :")
  for country in list_of_all_data_clusters[i]:
    mean_birth_rate += float(country[1])
    mean_life_expectancy += float(country[2])
    count += 1
  mean_birth_rate = round(mean_birth_rate / count, 2)
  mean_life_expectancy = round(mean_life_expectancy / count, 2)
  print(f"\tMean birth rate :{mean_birth_rate} \n\tMean life expectancy :{mean_life_expectancy}")


# Show plot
plt.legend()
plt.show()