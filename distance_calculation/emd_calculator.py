__author__ = 'GongLi'

from sklearn.cluster import KMeans
from pulp import *
import numpy as np


def emd_calculator_optimized_kmeans(feature1, feature2, distances=None, divide_by=3):
    """
    binned
    :param feature1: nd_array
    :param feature2: nd_array
    :return: emb
    """
    n_clusters1 = int(feature1.shape[0]/divide_by)
    n_clusters2 = int(feature2.shape[0]/divide_by)
    if feature1.shape[0] > n_clusters1:
        kmeans1 = KMeans(n_clusters1).fit(feature1)
        feature1 = kmeans1.cluster_centers_

        weights1 = np.zeros((feature1.shape[0],))
        for i in range(weights1.shape[0]):
            weights1[i] = np.sum(kmeans1.labels_ == i)
        f1_w = weights1/float(np.sum(weights1))
    else:
        f1_w = 1.0 / feature1.shape[0] * np.ones((feature1.shape[0],))

    if feature2.shape[0] > n_clusters2:
        kmeans2 = KMeans(n_clusters2).fit(feature2)
        feature2 = kmeans2.cluster_centers_
        weights2 = np.zeros((feature2.shape[0],))
        for i in range(weights2.shape[0]):
            weights2[i] = np.sum(kmeans2.labels_ == i)
        f2_w = weights2 / float(np.sum(weights2))
    else:
        f2_w = 1.0 / feature2.shape[0] * np.ones((feature2.shape[0],))

    H = feature1.shape[0]
    I = feature2.shape[0]

    if distances is None:
        distances = np.zeros((H, I))
        for i in range(H):
            for j in range(I):
                distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

    # Set variables for EMD calculations
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += lpSum(objectiveFunction)

    tempMin = 1.
    problem += lpSum(constraint) == tempMin

    # constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) <= f1_w[i]

    for j in range(I):
        constraint2 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) <= f2_w[j]

    # solve
    problem.writeLP("EMD.lp")
    problem.solve(GLPK_CMD())

    flow = value(problem.objective)

    return flow / tempMin


def emd_calculator_optimized(feature1, feature2, distances=None):
    """
    weights assigned uniformly to each data point
    :param feature1: nd_array
    :param feature2: nd_array
    :return: emb
    """
    H = feature1.shape[0]
    I = feature2.shape[0]

    f1_w = 1.0/H
    f2_w = 1.0/I

    if distances is None:
        distances = np.zeros((H, I))
        for i in range(H):
            for j in range(I):
                distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

    # Set variables for EMD calculations
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += lpSum(objectiveFunction)

    tempMin = 1.
    problem += lpSum(constraint) == tempMin

    # constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) <= f1_w

    for j in range(I):
        constraint2 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) <= f2_w

    # solve
    problem.writeLP("EMD.lp")
    problem.solve(GLPK_CMD())

    flow = value(problem.objective)

    return flow / tempMin


def emd_calculator(feature1, feature2, w1, w2):
    """

    :param feature1: nd_array
    :param feature2: nd_array
    :param w1: nd_array
    :param w2: nd_array
    :return: emb
    """
    H = feature1.shape[0]
    I = feature2.shape[0]

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

    # Set variables for EMD calculations
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += lpSum(objectiveFunction)


    tempMin = min(sum(w1), sum(w2))
    problem += lpSum(constraint) == tempMin

    # constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) <= w1[i]

    for j in range(I):
        constraint2 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) <= w2[j]

    # solve
    problem.writeLP("EMD.lp")
    problem.solve(GLPK_CMD())

    flow = value(problem.objective)


    return flow / tempMin


if __name__ == '__main__':
    feature1 = np.array([[1], [2], [3]])
    feature2 = np.array([[0], [1]])

    # w1_ = [0.4, 0.4, 0.4]
    # w2_ = [0.6, 0.6]

    # feature1 = np.array([[100, 40, 22], [211,20,2], [32, 190, 150], [ 2, 100, 100]])
    # feature2 = np.array([[0,0,0], [50, 100, 80], [255, 255, 255]])
    #
    # w1 = [0.4,0.3,0.2,0.1]
    # w2 = [0.5, 0.3, 0.2]
    #

    # emdDistance = emd_calculator(feature1, feature2, w1_, w2_)
    emdDistance = emd_calculator_optimized(feature1, feature2)
    print(str(emdDistance))
    emdDistance = emd_calculator_optimized_kmeans(feature1, feature2)
    print(str(emdDistance))
