import matplotlib.pyplot as plt
import pandas as pd



def plot_stats(filename):
    df = pd.read_csv(filename)

    # we choose as bin size the square root of the number of objects in our database sqrt(2483) ~= 50
    n_bin = 50
    vertices_mean = df['num_vertices'].mean()
    faces_mean = df['num_faces'].mean()
    vertices_median = df['num_vertices'].median()
    faces_median = df['num_faces'].median()

    print("vertices mean: "+str(vertices_mean))
    plt.figure()
    df['num_vertices'].hist(bins=n_bin)
    plt.axvline(vertices_mean, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean = {vertices_mean:.1f}')
    plt.axvline(vertices_median, color='green', linestyle='dotted', linewidth=2,
                label=f'Median = {vertices_median:.1f}')
    plt.xlabel("Number of vertices")
    plt.ylabel("Number of shapes")
    plt.title("Distribution of vertex counts")
    plt.legend()


    plt.figure()
    df['num_faces'].hist(bins=n_bin)
    plt.axvline(faces_mean, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean = {faces_mean:.1f}')
    plt.axvline(faces_median, color='green', linestyle='dotted', linewidth=2,
                label=f'Median = {faces_median:.1f}')
    plt.xlabel("Number of faces")
    plt.ylabel("Number of shapes")
    plt.title("Distribution of face counts")
    plt.legend()


    plt.figure(figsize=(20, 6))
    df['class'].value_counts().plot(kind='bar')
    plt.xlabel("Class")
    plt.ylabel("Number of shapes")
    plt.title("shapes per class")


    plt.show()