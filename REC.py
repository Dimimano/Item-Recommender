import numpy as np
import statistics

np.random.seed(1)
N=100
M=100
X=0.3
K=20
T=10

y = np.nextafter(10, 11)
#Basic Matrix, contains the 'known' ratings that will be used in order to predict the 'unknown' ratings.
matrix = np.random.uniform(1,y, size=(N,M))

adj_cosine_error_list = []
cosine_error_list = []
jaccard_error_list = []
dice_error_list = []

adj_cosine_error_list_weighted = []
cosine_error_list_weighted = []
jaccard_error_list_weighted = []
dice_error_list_weighted = []

adj_cosine_error_list_custom = []
adj_cosine_error_list_harmonic = []
cosine_error_list_custom = []
cosine_error_list_harmonic = []
jaccard_error_list_custom = []
jaccard_error_list_harmonic = []
dice_error_list_custom = []
dice_error_list_harmonic = []


while(K<=70):
    Amount_of_Test_Values = (X*M*N)
    for t in range(0,T):
        
        help_matrix = np.zeros((N,M))
        help_list = []
        counter = 0
        for i in range(0,N):
            for j in range(0,M):
                help_matrix[i][j] = counter
                help_list.append(counter)
                counter = counter + 1
        
        matrix_copy = matrix.copy()
        
        #Randomly choose which ratings to hide from the basic matrix.    
        X_values = np.random.choice(help_list, int(round(Amount_of_Test_Values)), replace=False)
        #Test Matrix, contains the 'unknown' ratings.
        test_matrix = np.zeros((N,M))
        for i in range(0,N):
            for j in range(0,M):
                if help_matrix[i][j] in X_values:
                    test_matrix[i][j] = matrix[i][j]
                    matrix[i][j] = 0
        
        #Adjused cosine
        #Matrix containing the average of every row, from the Basic matrix.
        average_Matrix = np.zeros(N)
        counter = 0
        row_sum = 0
        for i in range(0,N):
            for j in range(0,M):
                if matrix[i][j] == 0:
                    counter = counter + 1
                else:
                    row_sum = matrix[i][j] + row_sum
            average_Matrix[i] = row_sum/(N-counter)
            counter = 0
            row_sum = 0
        
        #Matrix adjusted to include every element of the basic matrix, after substracting the row's average.
        adjusted_Matrix = np.zeros((N,M))
        for i in range(0,N):
            for j in range(0,M):
                if matrix[i][j] != 0:
                    adjusted_Matrix[i][j] = matrix[i][j] - average_Matrix[i]
        
        #print(adjusted_Matrix)
        #Vector containing the L2 Norm for every column in the adjusted matrix.
        normVectorL2 = np.sqrt((adjusted_Matrix * adjusted_Matrix).sum(axis=0))
        adjusted_normalized_Matrix = np.zeros((N,M))
        
        #Matrix normalized using the L2 Norm.
        for i in range(0,N):
            for j in range(0,M):
                adjusted_normalized_Matrix[i][j] = adjusted_Matrix[i][j]/normVectorL2[j]
        
        #Summetric Matrix that contains the adjusted cosine similarity between each of the items.     
        adjusted_cosine = np.zeros((N,M))
        for i in range(0,M):
            for j in range(0,M):
                    adjusted_cosine[i][j] = np.dot(adjusted_normalized_Matrix[:,i],adjusted_normalized_Matrix[:,j])
               
        #Setting each value of the main diagonal to be 1. This process helps in order to later find the maximum similarity for every item.
        for i in range(0,M):
            for j in range(0,M):
                if i==j:
                    adjusted_cosine[i][j] = -1
                              
        adjusted_copy = adjusted_cosine.copy()
        
        #Matrix that contains in each column the item similarities in descending order.
        max_cosine = adjusted_cosine.max(1)
        #Matrix containing pointers to each item's most similar items.
        #For example, if item's 0 most similar item is item 2, then column's 0 first element is the value 2, pointing to that item.
        max_cosine_pointers = np.zeros((N-1,M))
        
        
        for i in range(0,M):
            for j in range(0,M):
                if adjusted_cosine[i][j] == max_cosine[i]:
                    adjusted_cosine[i][j] = -1
                    break        
        max_cosine2 = adjusted_cosine.max(1)
        max_cosine = np.vstack((max_cosine,max_cosine2))
        
        
        for k in range(0,M-3):
            for i in range(0,M):
                for j in range(0,M):
                    if adjusted_cosine[i][j] == max_cosine[max_cosine.shape[0]-1][i]:
                        adjusted_cosine[i][j] = -1
                        break
            max_cosine2 = adjusted_cosine.max(1)
            max_cosine = np.vstack((max_cosine,max_cosine2))
        
        #Create the pointers matrix. 
        for i in range(0,N-1):
            for j in range(0,M):
                maximum_cosine = max_cosine[i][j]
                for k in range(0,N):
                    if adjusted_copy[j][k]==maximum_cosine:
                        max_cosine_pointers[i][j] = k
                        adjusted_copy[j][k] = -1
                        break   
        
        
        matrix_predictions = np.zeros((N,M))
        matrix_predictions_normal = np.zeros((N,M))
        matrix_predictions_custom = np.zeros((N,M))
        matrix_predictions_harmonic = np.zeros((N,M))
        
        #predict using adjusted cosine
        for i in range(0,M):
            for j in range(0,N):
                #If any unknown rating is found...
                if matrix[i][j] == 0:
                    counter = 0
                    cosine_coefficients = []
                    ratings = []
                    flag = 0
                    k=0
                    #Using the preselected amount of neighbors...
                    while(k<K):
                        if counter < max_cosine_pointers.shape[0]:
                            pointer = int(max_cosine_pointers[counter][j])
                        elif counter >= max_cosine_pointers.shape[0]:
                            break
                        #While the most similar items have no ratings from the user, search for the next neighbor.
                        while matrix[i][pointer] == 0 and k < K:
                            if counter == max_cosine.shape[0]-1:
                                matrix_predictions[i][j] = -100
                                matrix_predictions_normal[i][j] = -100
                                matrix_predictions_custom[i][j] = -100
                                matrix_predictions_harmonic[i][j] = -100
                                flag = 1
                                break
                            elif counter < max_cosine.shape[0]-1:
                                counter = counter + 1
                                pointer = int(max_cosine_pointers[counter][j])
                                if int(matrix[i][pointer]) == 0:
                                    if counter < max_cosine.shape[0]-1:
                                        counter = counter + 1
                                        continue
                                    elif counter == max_cosine.shape[0]-1:
                                        flag = 1
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        break
                                else:
                                    ratings.append(matrix[i][pointer])                 
                                    cosine_coefficients.append(max_cosine[counter][j])
                                    counter = counter + 1
                                    if k == K-1:
                                        numerator = 0
                                        numerator2 = 0
                                        denominator = 0
                                        denominator2 = 0
                                        numerator3 = 0
                                        denominator3 = 0
                                        for x in range(0,K):                               
                                            numerator = cosine_coefficients[x]*ratings[x] + numerator
                                            denominator = cosine_coefficients[x] + denominator
                                            numerator2 = ratings[x] + numerator2
                                            denominator2 = 1 + denominator2
                                            mean=sum(ratings)/len(ratings)
                                            data = []
                                            if mean <= 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] <= 5.5:
                                                        numerator3 = ratings[x] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z])
                                                        
                                            elif mean > 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] > 5.5:
                                                        numerator3 = ratings[z] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z]) 
                                            if x == K-1 and denominator!=0:
                                                prediction = numerator/denominator
                                                prediction2 = numerator2/denominator2
                                                prediction3 = numerator3/denominator3
                                                prediction4 = statistics.harmonic_mean(data)
                                        matrix_predictions[i][j] = prediction
                                        matrix_predictions_normal[i][j] = prediction2
                                        matrix_predictions_custom[i][j] = prediction3
                                        matrix_predictions_harmonic[i][j] = prediction4
                                        
                                        flag = 1
                                    if k < K-1 and counter == max_cosine.shape[0]:
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        flag = 1
                                        break   
                                    k=k+1
                                     
                        if flag:
                            break
                        ratings.append(matrix[i][pointer])            
                        cosine_coefficients.append(max_cosine[counter][j])
                        counter = counter + 1
                        if k == K-1:
                            numerator = 0
                            numerator2 = 0
                            denominator = 0
                            denominator2 = 0
                            numerator3 = 0
                            denominator3 = 0
                            for x in range(0,K):                               
                                numerator = cosine_coefficients[x]*ratings[x] + numerator
                                denominator = cosine_coefficients[x] + denominator
                                numerator2 = ratings[x] + numerator2
                                denominator2 = 1 + denominator2
                                mean=sum(ratings)/len(ratings)
                                data = []
                                if mean <= 5.5:
                                    for z in range(0,K):
                                        if ratings[z] <= 5.5:
                                            numerator3 = ratings[x] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z])
                                                        
                                elif mean > 5.5:
                                    for z in range(0,K):
                                        if ratings[z] > 5.5:
                                            numerator3 = ratings[z] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z]) 
                                if x == K-1 and denominator!=0:
                                    prediction = numerator/denominator
                                    prediction2 = numerator2/denominator2
                                    prediction3 = numerator3/denominator3
                                    prediction4 = statistics.harmonic_mean(data)
                            matrix_predictions[i][j] = prediction
                            matrix_predictions_normal[i][j] = prediction2
                            matrix_predictions_custom[i][j] = prediction3
                            matrix_predictions_harmonic[i][j] = prediction4
                            break
                        k=k+1
                        
        #Calculate the Mean Absolute Error using the Weighted Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions[i][j] != 0 and matrix_predictions[i][j] != -100 and matrix_predictions[i][j] <= 11 and matrix_predictions[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions[i][j] - test_matrix[i][j]) + error_sum
        adj_cosine_error_list_weighted.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_normal[i][j] != 0 and matrix_predictions_normal[i][j] != -100 and matrix_predictions_normal[i][j] <= 11 and matrix_predictions_normal[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_normal[i][j] - test_matrix[i][j]) + error_sum     
        adj_cosine_error_list.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Harmonic Average of the (custom) predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_harmonic[i][j] != 0 and matrix_predictions_harmonic[i][j] != -100 and matrix_predictions_harmonic[i][j] <= 11 and matrix_predictions_harmonic[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_harmonic[i][j] - test_matrix[i][j]) + error_sum
                    
        adj_cosine_error_list_harmonic.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Custom Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_custom[i][j] != 0 and matrix_predictions_custom[i][j] != -100 and matrix_predictions_custom[i][j] <= 11 and matrix_predictions_custom[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_custom[i][j] - test_matrix[i][j]) + error_sum
                    
        adj_cosine_error_list_custom.append(error_sum/counter)

        
        #Cosine
        matrix_normalized = matrix.copy()
        normVectorL2 = np.sqrt((matrix_normalized * matrix_normalized).sum(axis=0))
        for i in range(0,N):
            for j in range(0,M):
                matrix_normalized[i][j] = matrix_normalized[i][j]/normVectorL2[j]
        
        cosine = np.zeros((N,M))
        for i in range(0,M):
            for j in range(0,M):
                    cosine[i][j] = np.dot(matrix_normalized[:,i],matrix_normalized[:,j])
        
        for i in range(0,M):
            for j in range(0,M):
                if i==j:
                    cosine[i][j] = -1
        cosine_copy = cosine.copy()
        max_cosine_normal = cosine.max(1)
                    
        
        for i in range(0,M):
            for j in range(0,M):
                if cosine[i][j] == max_cosine_normal[i]:
                    cosine[i][j] = -1
                    break      
        max_cosine_normal2 = cosine.max(1)
        max_cosine_normal = np.vstack((max_cosine_normal,max_cosine_normal2))
        
        
        for k in range(0,M-3):
            for i in range(0,M):
                for j in range(0,M):
                    if cosine[i][j] == max_cosine_normal[max_cosine_normal.shape[0]-1][i]:
                        cosine[i][j] = -1
                        break
            max_cosine_normal2 = cosine.max(1)
            max_cosine_normal = np.vstack((max_cosine_normal,max_cosine_normal2))
        
        max_cosine_pointers = np.zeros((N-1,M))
        for i in range(0,N-1):
            for j in range(0,M):
                maximum_cosine = max_cosine_normal[i][j]
                for k in range(0,N):
                    if cosine_copy[j][k]==maximum_cosine:
                        max_cosine_pointers[i][j] = k
                        cosine_copy[j][k] = -1
                        break

        
        matrix_predictions = np.zeros((N,M))
        matrix_predictions_normal = np.zeros((N,M))
        matrix_predictions_custom = np.zeros((N,M))
        matrix_predictions_harmonic = np.zeros((N,M))
        
        #Predict with cosine
        for i in range(0,M):
            for j in range(0,N):
                if matrix[i][j] == 0:
                    counter = 0
                    cosine_coefficients = []
                    ratings = []
                    flag = 0
                    k=0
                    while(k<K):
                        if counter < max_cosine_pointers.shape[0]:
                            pointer = int(max_cosine_pointers[counter][j])
                        elif counter >= max_cosine_pointers.shape[0]:
                            break
                        while matrix[i][pointer] == 0 and k < K:
                            if counter == max_cosine_normal.shape[0]-1:
                                matrix_predictions[i][j] = -100
                                matrix_predictions_normal[i][j] = -100
                                matrix_predictions_custom[i][j] = -100
                                matrix_predictions_harmonic[i][j] = -100
                                flag = 1
                                break
                            elif counter < max_cosine_normal.shape[0]-1:
                                counter = counter + 1
                                pointer = int(max_cosine_pointers[counter][j])
                                if int(matrix[i][pointer]) == 0:
                                    if counter < max_cosine_normal.shape[0]-1:
                                        counter = counter + 1
                                        continue
                                    elif counter == max_cosine_normal.shape[0]-1:
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        flag = 1
                                        break
                                else:
                                    ratings.append(matrix[i][pointer])                
                                    cosine_coefficients.append(max_cosine_normal[counter][j])
                                    counter = counter + 1
                                    if k == K-1:
                                        numerator = 0
                                        numerator2 = 0
                                        denominator = 0
                                        denominator2 = 0
                                        numerator3 = 0
                                        denominator3 = 0
                                        for x in range(0,K):                               
                                            numerator = cosine_coefficients[x]*ratings[x] + numerator
                                            denominator = cosine_coefficients[x] + denominator
                                            numerator2 = ratings[x] + numerator2
                                            denominator2 = 1 + denominator2
                                            mean=sum(ratings)/len(ratings)
                                            data = []  
                                            if mean <= 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] <= 5.5:
                                                        numerator3 = ratings[x] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z])
                                                        
                                            elif mean > 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] > 5.5:
                                                        numerator3 = ratings[z] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z])
                                                        
                                            if x == K-1 and denominator!=0:
                                                prediction = numerator/denominator
                                                prediction2 = numerator2/denominator2
                                                prediction3 = numerator3/denominator3
                                                prediction4 = statistics.harmonic_mean(data)
                                        matrix_predictions[i][j] = prediction
                                        matrix_predictions_normal[i][j] = prediction2
                                        matrix_predictions_custom[i][j] = prediction3
                                        matrix_predictions_harmonic[i][j] = prediction4
                                        flag = 1
                                    if k < K-1 and counter == max_cosine_normal.shape[0]:
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        flag = 1
                                        break   
                                    k=k+1
     
                        if flag:
                            break
                        ratings.append(matrix[i][pointer])                
                        cosine_coefficients.append(max_cosine[counter][j])
                        counter = counter + 1
                        if k == K-1:
                            numerator = 0
                            numerator2 = 0
                            denominator = 0
                            denominator2 = 0
                            numerator3 = 0
                            denominator3 = 0
                            for x in range(0,K):                               
                                numerator = cosine_coefficients[x]*ratings[x] + numerator
                                denominator = cosine_coefficients[x] + denominator
                                numerator2 = ratings[x] + numerator2
                                denominator2 = 1 + denominator2
                                mean=sum(ratings)/len(ratings)
                                data = []
                                if mean <= 5.5:
                                    for z in range(0,K):
                                        if ratings[z] <= 5.5:
                                            numerator3 = ratings[z] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z])
                                            
                                elif mean > 5.5:
                                    for z in range(0,K):
                                        if ratings[z] > 5.5:
                                            numerator3 = ratings[z] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z])
                        
                                
                                if x == K-1 and denominator!=0:
                                    prediction = numerator/denominator
                                    prediction2 = numerator2/denominator2
                                    prediction3 = numerator3/denominator3
                                    prediction4 = statistics.harmonic_mean(data)
                                    
                            matrix_predictions[i][j] = prediction
                            matrix_predictions_normal[i][j] = prediction2
                            matrix_predictions_custom[i][j] = prediction3
                            matrix_predictions_harmonic[i][j] = prediction4
                            #break
                        k=k+1
                        
        #Calculate the Mean Absolute Error using the Weighted Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions[i][j] != 0 and matrix_predictions[i][j] != -100 and matrix_predictions[i][j] <= 11 and matrix_predictions[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions[i][j] - test_matrix[i][j]) + error_sum
        cosine_error_list_weighted.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_normal[i][j] != 0 and matrix_predictions_normal[i][j] != -100 and matrix_predictions_normal[i][j] <= 11 and matrix_predictions_normal[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_normal[i][j] - test_matrix[i][j]) + error_sum
                    
        cosine_error_list.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Harmonic Average of the (custom) predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_harmonic[i][j] != 0 and matrix_predictions_harmonic[i][j] != -100 and matrix_predictions_harmonic[i][j] <= 11 and matrix_predictions_harmonic[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_harmonic[i][j] - test_matrix[i][j]) + error_sum
                    
        cosine_error_list_harmonic.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Custom Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_custom[i][j] != 0 and matrix_predictions_custom[i][j] != -100 and matrix_predictions_custom[i][j] <= 11 and matrix_predictions_custom[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_custom[i][j] - test_matrix[i][j]) + error_sum
                    
        cosine_error_list_custom.append(error_sum/counter)
        
        #Jaccard & Dice
        jaccard = np.zeros((N,M))
        dice = np.zeros((N,M))
        
        #Calculate Jaccard and Dice into symmetric Matrix.
        for k in range(0,M):
            for i in range(0,M):
                intersection = 0
                union = 0
                for j in range(0,N):
                    #If there is a rating for each of 2 items..
                    if matrix[j][k] != 0 and matrix[j][i] != 0:
                        #..increase both the union and the intersection.
                        intersection = intersection + 1
                        union = union + 1
                    #If there is a rating only for one of the items..
                    elif matrix[j][k] == 0 and matrix[j][i] != 0:
                        #..increase only the union.
                        union = union + 1
                    #If there is a rating only for one of the items..
                    elif matrix[j][k] != 0 and matrix[j][i] == 0:
                        #..increase only the union.
                        union = union + 1 
                    #If the last rating is reached calculate both the Jaccard & Dice similarities.
                    if j == N-1:
                        jaccard[k][i] = intersection/union 
                        dice[k][i] = jaccard[k][i]*2/(1+jaccard[k][i])
        
        
        for i in range(0,M):
            for j in range(0,M):
                if i==j:
                    jaccard[i][j] = -1
                    dice[i][j] = -1
        jaccard_copy = jaccard.copy()
        dice_copy = dice.copy()
        max_jaccard = jaccard.max(1)
        max_jaccard_pointers = np.zeros((N-1,M))
        max_dice = dice.max(1)
        max_dice_pointers = np.zeros((N-1,M))
        
        
        for i in range(0,M):
            for j in range(0,M):
                if jaccard[i][j] == max_jaccard[i]:
                    jaccard[i][j] = -1
                    break
                
        for i in range(0,M):
            for j in range(0,M):
                if dice[i][j] == max_dice[i]:
                    dice[i][j] = -1
                    break
    
        max_dice2 = dice.max(1)
        max_dice = np.vstack((max_dice,max_dice2))
        max_jaccard2 = jaccard.max(1)
        max_jaccard = np.vstack((max_jaccard,max_jaccard2))
        
        for k in range(0,M-3):
            for i in range(0,M):
                for j in range(0,M):
                    if jaccard[i][j] == max_jaccard[max_jaccard.shape[0]-1][i]:
                        jaccard[i][j] = -1
                        break
            max_jaccard2 = jaccard.max(1)
            max_jaccard = np.vstack((max_jaccard,max_jaccard2))
        
        for i in range(0,N-1):
            for j in range(0,M):
                maximum_jaccard = max_jaccard[i][j]
                for k in range(0,N):
                    if jaccard_copy[j][k]==maximum_jaccard:
                        max_jaccard_pointers[i][j] = k
                        jaccard_copy[j][k] = -1
                        break
    
        for k in range(0,M-3):
            for i in range(0,M):
                for j in range(0,M):
                    if dice[i][j] == max_dice[max_dice.shape[0]-1][i]:
                        dice[i][j] = -1
                        break
            max_dice2 = dice.max(1)
            max_dice = np.vstack((max_dice,max_dice2))
        
        for i in range(0,N-1):
            for j in range(0,M):
                maximum_dice = max_dice[i][j]
                for k in range(0,N):
                    if dice_copy[j][k]==maximum_dice:
                        max_dice_pointers[i][j] = k
                        dice_copy[j][k] = -1
                        break

        matrix_predictions = np.zeros((N,M))
        matrix_predictions_normal = np.zeros((N,M))
        matrix_predictions_custom = np.zeros((N,M))
        matrix_predictions_harmonic = np.zeros((N,M))
        
        #predict with Jaccard
        for i in range(0,M):
            for j in range(0,N):
                if matrix[i][j] == 0:
                    counter = 0
                    jaccard_coefficients = []
                    ratings = []
                    flag = 0
                    k=0
                    while(k<K):
                        if counter < max_jaccard_pointers.shape[0]:
                            pointer = int(max_jaccard_pointers[counter][j])
                        elif counter >= max_jaccard_pointers.shape[0]:
                            break
                        while matrix[i][pointer] == 0 and k < K:
                            if counter == max_jaccard.shape[0]-1:
                                matrix_predictions[i][j] = -100
                                matrix_predictions_normal[i][j] = -100
                                matrix_predictions_custom[i][j] = -100
                                matrix_predictions_harmonic[i][j] = -100
                                flag = 1
                                break
                            elif counter < max_jaccard.shape[0]-1:
                                counter = counter + 1
                                pointer = int(max_jaccard_pointers[counter][j])
                                if int(matrix[i][pointer]) == 0:
                                    if counter < max_jaccard.shape[0]-1:
                                        counter = counter + 1
                                        continue
                                    elif counter == max_jaccard.shape[0]-1:
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        flag = 1
                                        break
                                else:
                                    ratings.append(matrix[i][pointer])                
                                    jaccard_coefficients.append(max_jaccard[counter][j])
                                    counter = counter + 1
                                    if k == K-1:
                                        numerator = 0
                                        numerator2 = 0
                                        denominator = 0
                                        denominator2 = 0
                                        numerator3 = 0
                                        denominator3 = 0
                                        for x in range(0,K):                               
                                            numerator = jaccard_coefficients[x]*ratings[x] + numerator
                                            denominator = jaccard_coefficients[x] + denominator
                                            numerator2 = ratings[x] + numerator2
                                            denominator2 = 1 + denominator2
                                            mean=sum(ratings)/len(ratings)
                                            data = []
                                            if mean <= 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] <= 5.5:
                                                        numerator3 = ratings[x] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z])
                                                        
                                            elif mean > 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] > 5.5:
                                                        numerator3 = ratings[z] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z])
                                                          
                                            if x == K-1 and denominator!=0:
                                                prediction = numerator/denominator
                                                prediction2 = numerator2/denominator2
                                                prediction3 = numerator3/denominator3
                                                prediction4 = statistics.harmonic_mean(data)
                                            
                                        matrix_predictions[i][j] = prediction
                                        matrix_predictions_normal[i][j] = prediction2
                                        matrix_predictions_custom[i][j] = prediction3
                                        matrix_predictions_harmonic[i][j] = prediction4
                                        flag = 1
                                        break
                                    if k < K-1 and counter == max_jaccard.shape[0]:
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        flag = 1
                                        break   
                                    k=k+1     
                        if flag:
                            break
                        ratings.append(matrix[i][pointer])          
                        jaccard_coefficients.append(max_jaccard[counter][j])
                        counter = counter + 1
                        if k == K-1:
                            numerator = 0
                            numerator2 = 0
                            denominator = 0
                            denominator2 = 0
                            numerator3 = 0
                            denominator3 = 0
                            for x in range(0,K):                               
                                numerator = jaccard_coefficients[x]*ratings[x] + numerator
                                denominator = jaccard_coefficients[x] + denominator
                                numerator2 = ratings[x] + numerator2
                                denominator2 = 1 + denominator2
                                mean=sum(ratings)/len(ratings)
                                data = []
                                if mean <= 5.5:
                                    for z in range(0,K):
                                        if ratings[z] <= 5.5:
                                            numerator3 = ratings[x] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z])
                                                        
                                elif mean > 5.5:
                                    for z in range(0,K):
                                        if ratings[z] > 5.5:
                                            numerator3 = ratings[z] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z])
                                if x == K-1 and denominator!=0:
                                    prediction = numerator/denominator
                                    prediction2 = numerator2/denominator2
                                    prediction3 = numerator3/denominator3
                                    prediction4 = statistics.harmonic_mean(data)
                            matrix_predictions[i][j] = prediction
                            matrix_predictions_normal[i][j] = prediction2
                            matrix_predictions_custom[i][j] = prediction3
                            matrix_predictions_harmonic[i][j] = prediction4
                            break
                        k=k+1
        
        #Calculate the Mean Absolute Error using the Weighted Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions[i][j] != 0 and matrix_predictions[i][j] != -100 and matrix_predictions[i][j] <= 11 and matrix_predictions[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions[i][j] - test_matrix[i][j]) + error_sum
        jaccard_error_list_weighted.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_normal[i][j] != 0 and matrix_predictions_normal[i][j] != -100 and matrix_predictions_normal[i][j] <= 11 and matrix_predictions_normal[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_normal[i][j] - test_matrix[i][j]) + error_sum
                    
        jaccard_error_list.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Harmonic Average of the (custom) predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_harmonic[i][j] != 0 and matrix_predictions_harmonic[i][j] != -100 and matrix_predictions_harmonic[i][j] <= 11 and matrix_predictions_harmonic[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_harmonic[i][j] - test_matrix[i][j]) + error_sum
                    
        jaccard_error_list_harmonic.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Custom Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_custom[i][j] != 0 and matrix_predictions_custom[i][j] != -100 and matrix_predictions_custom[i][j] <= 11 and matrix_predictions_custom[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_custom[i][j] - test_matrix[i][j]) + error_sum
                    
        jaccard_error_list_custom.append(error_sum/counter)
        
        matrix_predictions = np.zeros((N,M))
        matrix_predictions_normal = np.zeros((N,M))
        matrix_predictions_custom = np.zeros((N,M))
        matrix_predictions_harmonic = np.zeros((N,M))
        
        #Predict with Dice
        for i in range(0,M):
            for j in range(0,N):
                if matrix[i][j] == 0:
                    counter = 0
                    dice_coefficients = []
                    ratings = []
                    flag = 0
                    k=0
                    while(k<K):
                        if counter < max_dice_pointers.shape[0]:
                            pointer = int(max_dice_pointers[counter][j])
                        elif counter >= max_dice_pointers.shape[0]:
                            break
                        while matrix[i][pointer] == 0 and k < K:
                            if counter == max_dice.shape[0]-1:
                                matrix_predictions[i][j] = -100
                                matrix_predictions_normal[i][j] = -100
                                matrix_predictions_custom[i][j] = -100
                                matrix_predictions_harmonic[i][j] = -100
                                flag = 1
                                break
                            elif counter < max_dice.shape[0]-1:
                                counter = counter + 1
                                pointer = int(max_dice_pointers[counter][j])
                                if int(matrix[i][pointer]) == 0:
                                    if counter < max_dice.shape[0]-1:
                                        counter = counter + 1
                                        continue
                                    elif counter == max_dice.shape[0]-1:
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        flag = 1
                                        break
                                else:
                                    ratings.append(matrix[i][pointer])                
                                    dice_coefficients.append(max_dice[counter][j])
                                    counter = counter + 1
                                    if k == K-1:
                                        numerator = 0
                                        numerator2 = 0
                                        denominator = 0
                                        denominator2 = 0
                                        numerator3 = 0
                                        denominator3 = 0
                                        for x in range(0,K):                               
                                            numerator = dice_coefficients[x]*ratings[x] + numerator
                                            denominator = dice_coefficients[x] + denominator
                                            numerator2 = ratings[x] + numerator2
                                            denominator2 = 1 + denominator2
                                            mean=sum(ratings)/len(ratings)
                                            data = []
                                            if mean <= 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] <= 5.5:
                                                        numerator3 = ratings[x] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z])
                                                        
                                            elif mean > 5.5:
                                                for z in range(0,K):
                                                    if ratings[z] > 5.5:
                                                        numerator3 = ratings[z] + numerator3
                                                        denominator3 = 1 + denominator3
                                                        data.append(ratings[z])
                                            if x == K-1 and denominator!=0:
                                                prediction = numerator/denominator
                                                prediction2 = numerator2/denominator2
                                                prediction3 = numerator3/denominator3
                                                prediction4 = statistics.harmonic_mean(data)
                                        matrix_predictions[i][j] = prediction
                                        matrix_predictions_normal[i][j] = prediction2
                                        matrix_predictions_custom[i][j] = prediction3
                                        matrix_predictions_harmonic[i][j] = prediction4
                                        flag = 1
                                        break
                                    if k < K-1 and counter == max_dice.shape[0]:
                                        matrix_predictions[i][j] = -100
                                        matrix_predictions_normal[i][j] = -100
                                        matrix_predictions_custom[i][j] = -100
                                        matrix_predictions_harmonic[i][j] = -100
                                        flag = 1
                                        break   
                                    k=k+1     
                        if flag:
                            break
                        ratings.append(matrix[i][pointer])            
                        dice_coefficients.append(max_dice[counter][j])
                        counter = counter + 1
                        if k == K-1:
                            numerator = 0
                            numerator2 = 0
                            denominator = 0
                            denominator2 = 0
                            numerator3 = 0
                            denominator3 = 0
                            for x in range(0,K):                               
                                numerator = dice_coefficients[x]*ratings[x] + numerator
                                denominator = dice_coefficients[x] + denominator
                                numerator2 = ratings[x] + numerator2
                                denominator2 = 1 + denominator2
                                mean=sum(ratings)/len(ratings)
                                data = []
                                if mean <= 5.5:
                                    for z in range(0,K):
                                        if ratings[z] <= 5.5:
                                            numerator3 = ratings[x] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z])
                                                        
                                elif mean > 5.5:
                                    for z in range(0,K):
                                        if ratings[z] > 5.5:
                                            numerator3 = ratings[z] + numerator3
                                            denominator3 = 1 + denominator3
                                            data.append(ratings[z])
                                
                                if x == K-1 and denominator!=0:
                                    prediction = numerator/denominator
                                    prediction2 = numerator2/denominator2
                                    prediction3 = numerator3/denominator3
                                    prediction4 = statistics.harmonic_mean(data)
                            matrix_predictions[i][j] = prediction
                            matrix_predictions_normal[i][j] = prediction2
                            matrix_predictions_custom[i][j] = prediction3
                            matrix_predictions_harmonic[i][j] = prediction4
                            break
                        k=k+1
        
        #Calculate the Mean Absolute Error using the Weighted Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions[i][j] != 0 and matrix_predictions[i][j] != -100 and matrix_predictions[i][j] <= 11 and matrix_predictions[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions[i][j] - test_matrix[i][j]) + error_sum
        dice_error_list_weighted.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_normal[i][j] != 0 and matrix_predictions_normal[i][j] != -100 and matrix_predictions_normal[i][j] <= 11 and matrix_predictions_normal[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_normal[i][j] - test_matrix[i][j]) + error_sum
                    
        dice_error_list.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Harmonic Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_harmonic[i][j] != 0 and matrix_predictions_harmonic[i][j] != -100 and matrix_predictions_harmonic[i][j] <= 11 and matrix_predictions_harmonic[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_harmonic[i][j] - test_matrix[i][j]) + error_sum
                    
        dice_error_list_harmonic.append(error_sum/counter)
        
        #Calculate the Mean Absolute Error using the Custom Average of the predicted ratings.
        error_sum = 0
        counter = 0
        for i in range(0,M):
            for j in range(0,N):
                if  matrix_predictions_custom[i][j] != 0 and matrix_predictions_custom[i][j] != -100 and matrix_predictions_custom[i][j] <= 11 and matrix_predictions_custom[i][j] > 1:
                    counter = counter + 1
                    error_sum = abs(matrix_predictions_custom[i][j] - test_matrix[i][j]) + error_sum
                    
        dice_error_list_custom.append(error_sum/counter)
        
        matrix = matrix_copy
    
    print(K)    
    print("Adjusted Cosine Weighted MAE: ", sum(adj_cosine_error_list_weighted)/len(adj_cosine_error_list_weighted))
    print("Adjusted Cosine MAE: ", sum(adj_cosine_error_list)/len(adj_cosine_error_list))
    print("Cosine Weighted MAE: ", sum(cosine_error_list_weighted)/len(cosine_error_list_weighted))
    print("Cosine MAE: ", sum(cosine_error_list)/len(cosine_error_list))
    print("Jaccard Weighted MAE: ", sum(jaccard_error_list_weighted)/len(jaccard_error_list_weighted))
    print("Jaccard MAE: ", sum(jaccard_error_list)/len(jaccard_error_list))
    print("Dice Weighted MAE: ", sum(dice_error_list_weighted)/len(dice_error_list_weighted))
    print("Dice MAE: ", sum(dice_error_list)/len(dice_error_list))
    print("Adjusted Cosine Custom MAE: ", sum(adj_cosine_error_list_custom)/len(adj_cosine_error_list_custom))
    print("Adjusted Cosine Harmonic MAE: ", sum(adj_cosine_error_list_harmonic)/len(adj_cosine_error_list_harmonic))
    print("Cosine Custom MAE: ", sum(cosine_error_list_custom)/len(cosine_error_list_custom))
    print("Cosine Harmonic MAE: ", sum(cosine_error_list_harmonic)/len(cosine_error_list_harmonic))
    print("Jaccard Custom MAE: ", sum(jaccard_error_list_custom)/len(jaccard_error_list_custom))
    print("Jaccard Harmonic MAE: ", sum(jaccard_error_list_harmonic)/len(jaccard_error_list_harmonic))
    print("Dice Custom MAE: ", sum(dice_error_list_custom)/len(dice_error_list_custom))
    print("Dice Harmonic MAE: ", sum(dice_error_list_harmonic)/len(dice_error_list_harmonic))
    print(" ")
    if K < 10:
        K=K+4
    else:
        K=K+10
    


