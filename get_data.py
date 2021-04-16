import requests
import pandas as pd
import json
import urllib.request as urllib
import zipfile
import os
import time

def get_data(folder="data"):
    df_stop_search = get_stop_search()
    df_simd = get_simd()
    df_wards = get_zones_wards(df_simd)
    return df_stop_search, df_simd, df_wards

def get_stop_search(folder="data"):
    url = "https://www.scotland.police.uk/spa-media/pssjskv1/1b-01-april-31-december-2020-csv.csv"
    r = requests.get(url)
    decoded_content = r.content.decode('utf-8')
    csv_file_location = os.path.join(folder,"stop_search.csv")
    csvfile = open(csv_file_location, "w")
    csvfile.write(decoded_content)
    csvfile.close()
    df_stop_search = pd.read_csv(csv_file_location)
    return df_stop_search

def get_simd(folder="data"):
    #download the zip file from the simd website
    url = ("https://simd.scot/data/simd2020_withgeog.zip")  
    filehandle, _ = urllib.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    #egt the simd csv file from the archive
    simd_file_name = zip_file_object.namelist()[2]
    #open the file
    simd_file = zip_file_object.open(simd_file_name)
    df_simd = pd.read_csv(simd_file)
    df_simd.to_csv(os.path.join(folder,"simd_2020.csv"))
    return df_simd

def geog_url(code):
    return f"https://statistics.gov.scot/resource.json?uri=http%3A%2F%2Fstatistics.gov.scot%2Fid%2Fstatistical-geography%2F{code}"

def get_zones_wards(df_simd, folder="data", rate_limit=1/15):
    df_wards = pd.DataFrame(columns = ['zone_code' , 'zone_name', 'ward_code' , 'ward_name'])
    i = 0
    for zone_code in df_simd["Data_Zone"]:
        time.sleep(rate_limit)
        r = requests.get(geog_url(zone_code))
        zone_data = json.loads(r.text)[0]
        zone_name = zone_data["http://www.w3.org/2000/01/rdf-schema#label"][0]["@value"]
        ward_code = zone_data["http://statistics.gov.scot/def/hierarchy/best-fit#electoral-ward"][0]["@id"].split("/")[-1]
        
        #get the name of the ward
        time.sleep(rate_limit)
        r = requests.get(geog_url(ward_code))
        json.loads(r.text)[0]
        
        ward_data = json.loads(r.text)[0]
        ward_name = ward_data["http://www.w3.org/2000/01/rdf-schema#label"][0]["@value"]

        df_wards = df_wards.append({"zone_code":zone_code, "zone_name":zone_name, "ward_code":ward_code, "ward_name":ward_name}, ignore_index=True)
        i+=1
        print(i)

    df_wards.to_csv(os.path.join(folder,"wards.csv"))
    return df_wards

if __name__ == "__main__":

    df_stop_search, df_simd, df_wards = get_data()

