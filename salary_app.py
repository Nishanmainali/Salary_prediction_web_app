import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import re
from custom_transformers import MultiLabelBinarizerDF

 

# Load pipeline 

with open("salary_pipeline.pkl", "rb") as f: 
    model = pickle.load(f) 

st.title("💼 Salary Prediction App") 

 # User Inputs 

workmode = st.selectbox("Work Mode", ["Remote", "Hybrid", "In-person"]) 

edlevel = st.selectbox("Education Level", ["Bachelor's" ," Master's" , "Some College", "Other" , "Advanced Degree"]) 

devtype = st.selectbox("Developer Type", ['Front-End Developer', 'Full-Stack Developer', 

                        'Back-End Developer', 'System Administrator', 'Designer/UX', 

                        'Security', 'developer, embedded applications or devices', 

                        'engineer, data', 'Project/Product Manager', 

                        'developer, desktop or enterprise applications', 

                        'Executive / Manager', 'Other', 'developer, qa or test', 

                        'Developer Advocate', 'Data Scientist / ML', 'DevOps / SRE', 

                        'developer, mobile', 'Data Analyst', 'Academic/Research', 

                        'Blockchain Developer', 'engineer, site reliability', 

                        'developer, game or graphics', 'Embedded/IoT Developer', 

                        'Marketing/Sales']) 

orgsize = st.selectbox("Organization Size", ['Medium(100 to 499)', 'Small(2 to 99)', 'Large(1,000 to 4,999)', 

                        'Freelancer', 'Enterprise(More than 4,999)', 'I don’t know']) 

country = st.selectbox("Country", ['United States of America', 'Other', 

                        'United Kingdom of Great Britain and Northern Ireland', 'Finland', 

                        'Australia', 'Germany', 'Sweden', 'France', 'Spain', 'Brazil', 

                        'Italy', 'Canada', 'Switzerland', 'Netherlands', 'India', 'Norway', 

                        'Poland', 'Portugal', 'Austria', 'Denmark']) 

experience = st.slider("Years of Experience", min_value=0.0, max_value=51.0, step=1.0) 

language = st.multiselect("Languages Known", ['Bash/Shell (all shells)', 'C', 'C#', 'C++', 'Dart', 'Go',  

                        'Groovy', 'HTML/CSS', 'Java', 'JavaScript', 'Kotlin', 'Lua',  

                        'Other', 'PHP', 'PowerShell', 'Python', 'Ruby', 'Rust', 'SQL',  'Swift', 'TypeScript']) 

database = st.multiselect("Databases Used", ['BigQuery', 'Cassandra', 'Cloud Firestore', 'Cosmos DB', 'Dynamodb', 

                        'Elasticsearch', 'Firebase Realtime Database', 'H2', 'InfluxDB',  

                        'MariaDB', 'Microsoft Access', 'Microsoft SQL Server',  

                        'MongoDB', 'MySQL', 'Oracle', 'Other', 'PostgreSQL', 'Redis', 'SQLite', 'Snowflake', 'Supabase']) 

platform = st.multiselect("Platforms Used", ['Amazon Web Services (AWS)', 'Cloudflare', 'Digital Ocean', 'Firebase', 

                            'Fly.io', 'Google Cloud', 'Heroku', 'Hetzner', 'Linode, now Akamai',  

                            'Managed Hosting', 'Microsoft Azure', 'Netlify', 'OVH', 'OpenShift',  

                            'OpenStack', 'Oracle Cloud Infrastructure (OCI)', 'Other', 'Render', 

                            'VMware', 'Vercel', 'Vultr']) 

 

# Combine into DataFrame 

input_df = pd.DataFrame([{ 

"WorkMode": workmode, 

"EdLevel": edlevel, 

"DevType": devtype, 

"OrgSize": orgsize, 

"Country": country, 

"Experience": experience, 

"Language": "; ".join(language), 

"DataBase": "; ".join(database), 

"Platform": "; ".join(platform) 

}]) 

 

# Prediction 

if st.button("Predict Salary"): 
    salary = model.predict(input_df)[0] 
    st.success(f"💰 Predicted Salary: ${salary:,.2f}") 

