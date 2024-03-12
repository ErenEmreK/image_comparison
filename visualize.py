from image_compare import euclidean_distance, cosine_similarity_score
import os
import pandas as pd
import re

folder = 'perspective' #folder of images to compare
cos_excel_file = 'pers_cos_graph.xlsx' #excel file locations
euc_excel_file = 'pers_euc_graph.xlsx'

def extract_number(file):
    #sorting via numbers in image files
    match = re.search(r'\d+', file)
    return int(match.group()) if match else float('inf')

def table_maker(folder): 
    #Returns 2D tables for euc and cos scores.
    image_table = []
    images = os.listdir(folder)
 
            
    for image in images:
        if (image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png")):
            image_table.append(image)
            
            
    image_table.sort(key = extract_number)
    length = len(image_table)
    
    cosine_table = [[0 for _ in range(length)] for _ in range(length)]
    euclidean_table = [[0 for _ in range(length)] for _ in range(length)]

            
    for i in range(length):
        for j in range(length):
            image1 = folder + "/" + image_table[i]
            image2 = folder + "/" + image_table[j]
              
            cosine_table[i][j] = round(cosine_similarity_score(image1, image2), 2)
            euclidean_table[i][j] = round(euclidean_distance(image1, image2), 2)

    return cosine_table, euclidean_table  
            
def name_changer():
    #It cahnges the names of images to match indexes in case it's helpful
    #Nothing crucial here
    images = os.listdir(folder)
    count = 0
       
    for image in images:
        new_name = "image" + str(count) + ".jpg"
        os.rename(folder + "/" + image, folder + "/" + new_name)
        count += 1
    print("Files are renamed!")
    
    return

def main():
    cosine_table, euclidean_table = table_maker(folder)

    # Creates a DataFrame from the 2D array
    cos_df = pd.DataFrame(cosine_table)
    euc_df = pd.DataFrame(euclidean_table)

    # Converts to excel file
    cos_df.to_excel(cos_excel_file)
    euc_df.to_excel(euc_excel_file)

    print("Excel files are ready.")

if __name__ == "__main__":
    main()
