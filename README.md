import os
from glob import glob
from itertools import zip_longest

# 🔁 Replace with your actual parser function
def parse_multiple_xml_etree(file_path):
    # This is just a placeholder
    return [
        ({
            'ClaimID': ['001'], 
            'PatientName': ['John Doe']
        }, ['ClaimID', 'PatientName'])
    ]

# 📁 Folder containing XML files
folder_path = r'C:\path\to\xml\folder'  # <-- change this
output_file = 'merged_output.txt'

all_data_rows = []
header_written = False
final_columns = []

# 🔁 Process each XML file
for file_path in glob(os.path.join(folder_path, "*.xml")):
    print(f"📄 Parsing: {os.path.basename(file_path)}")
    
    parsed_rows = parse_multiple_xml_etree(file_path)
    
    for result, tag_order in parsed_rows:
        if not header_written:
            final_columns = tag_order
            header_written = True
        
        # Prepare rows (aligned by columns)
        max_len = max((len(v) for v in result.values()), default=1)
        row_data = {col: result.get(col, [""] * max_len) for col in final_columns}
        rows = list(zip_longest(*[row_data[col] for col in final_columns], fillvalue=""))
        all_data_rows.extend(rows)

# 📝 Write to text file (tab-separated)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\t'.join(final_columns) + '\n')
    for row in all_data_rows:
        f.write('\t'.join(str(val) for val in row) + '\n')

print(f"\n✅ All XML data merged and written to: {output_file}")











import pyodbc
import pandas as pd

# Define connection
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=172.28.196.3,1433;'
    'DATABASE=PowerSTEPP;'
    'Trusted_Connection=yes;'
)

# Table to export
table_name = 'dbo.Employees'  # Replace with your actual table name

# Read entire table into DataFrame
query = f"SELECT * FROM {table_name}"
df = pd.read_sql(query, conn)

# Export to Excel
output_path = 'employees_data.xlsx'  # Change file name/path as needed
df.to_excel(output_path, index=False)

print(f"✅ Data exported to Excel: {output_path}")

conn.close()








import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=172.28.196.3,1433;'
    'DATABASE=PowerSTEPP;'
    'Trusted_Connection=yes;'
)

cursor = conn.cursor()

# Get all table names
cursor.execute("""
    SELECT TABLE_SCHEMA, TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE'
""")

print("📋 Tables in PowerSTEPP database:")
for row in cursor.fetchall():
    print(f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}")

cursor.close()
conn.close()








import pyodbc

# Define connection string
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=MHP01HEALTHEDG;'
    'DATABASE=your_database_name;'
    'UID=your_username;'
    'PWD=your_password'
)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sys.databases")

for row in cursor.fetchall():
    print("📁", row[0])

conn.close()










ChatGPT
Saved memory full
You said:
This is my resume i need to answer about me in 900 characters

● 4 years of experience in Deep learning, Computer vision and Image processing software design 
and development for desktop and embedded environments
▶ Experience in structural and object-oriented programming.
▶ Experience in Supervised and Un-Supervised Learning.
▶ Experience in working with ARMNN, Snapdragon, Advantech, Intel NUC, neural compute stick,
Thermal sensor with Raspberry-pi.
▶ Experience in RAG with LLM’s and using opensource models.
▶ Experience with AWS, Azure and few other popular API based LLM invoke Mechanisms.
▶ Having knowledge on Generative AI, Active Learning, Knowledge Distillation, Pruning, Hierarchal
learning, NLP and Curriculum Learning.
▶ Experience in porting the Deep learning models to light weight platforms. (Micro
Processors/Micro Controllers)
▶ Experience in Data Clustering Techniques.
▶ Experience with Lidar data for object detection and classification along with lane segmentation.
▶ Experience with Medical Imaging like cell classification, x-ray enhancement, cell segmentation.
▶ Having Knowledge on DETR where Transformers can be used for detecting objects in an image.
Also other architectures like HDRNET, R2CNN.
▶ Experience in developing image processing algorithms in python, Matlab(beginner).
▶ Experience in developing computer vision solutions using OpenCV,Scikit-image, Matplotlib.
▶ Design and develop deep learning algorithms using Keras, Tensorflow, TF-Lite, Pytorch,OPENVINO
▶ Experience in developing deep learning application for object detection using YOLO, RESNET,
Mobilenet, Squeezenet, Peleenet and custom designed DL Model and so on.
▶ Experience in using Segmentation architectures like U-Net,E-net,ResUnet.
▶ Experience in using LLAMA, GPT and other LLM models through fastchat and RAG enabling
seamless communication with PDF and other documentations.
▶ Experience in Object Tracking and Monitoring and worked on Homograph for multiple cameras
feed.
▶ Experience with FFMPEG, Gstreamer, Yocto Linux platforms.
▶ Experience in developing deep learning application for person reidentification using siamese
neural networks.
▶ Experience in developing deep learning application for single shot object recognition using AI
model trained using triplet loss.
▶ Experience in developing deep learning application for sequential data using LSTM(RNNS),
Transformers.
▶ Experience in using Auto encoders for identification of defects in the image.
▶ Knowledge of using GANs for semantic segmentation and also had experiencing in rewriting GAN
for novelty features.
▶ Experience in using Multi task loss and Active learning for improving the accuracy of Deep
ChatGPT said:
Here is a concise and impactful "About Me" response within 900 characters, based on your resume:

I’m a Deep Learning and Computer Vision Engineer with 4 years of experience building and deploying AI solutions across embedded, desktop, and cloud platforms. I specialize in model optimization and porting DL models to edge devices like ARMNN, Intel NUC, and Raspberry Pi. I’ve worked extensively with YOLO, U-Net, ResNet, and transformers like DETR for object detection, segmentation, and classification—including Lidar and medical imaging use cases. I have hands-on experience with LLMs and RAG pipelines using open-source models, FastChat, AWS, and Azure. My strengths include Generative AI, Active Learning, and real-time video analytics using FFMPEG/GStreamer. I bring a strong foundation in both traditional image processing and deep learning, with a focus on efficient, scalable AI applications.














import pandas as pd
from openpyxl import load_workbook

# Step 1: Load the source Excel
file_path = 'your_file.xlsx'
sheet1_df = pd.read_excel(file_path, sheet_name='Sheet1')

# Step 2: Copy column structure for the new sheet
prd_sample_df = pd.DataFrame(columns=sheet1_df.columns)

# Step 3: Apply transformations to a temporary copy
temp_df = sheet1_df.copy()

# Format Correction: Fix ClaimType
temp_df['ClaimType'] = temp_df['ClaimType'].replace({
    'Institutional Inpatient': 'Institutional',
    'Institutional Outpatient': 'Institutional'
})

# Format Correction: Fix POS (prefix 00)
temp_df['POS'] = temp_df['POS'].apply(lambda x: f"{int(x):04}" if pd.notnull(x) else x)

# Step 4: Define your column mapping (from Sheet1 → prd_sample)
column_mapping = {
    'ClaimType': 'ClaimType',
    'POS': 'POS'
}

# Step 5: Populate values into prd_sample_df using the mapping
for src_col, target_col in column_mapping.items():
    if src_col in temp_df.columns and target_col in prd_sample_df.columns:
        prd_sample_df[target_col] = temp_df[src_col]

# Step 6: Write both sheets into a new Excel file
with pd.ExcelWriter('updated_output.xlsx', engine='openpyxl', mode='w') as writer:
    sheet1_df.to_excel(writer, sheet_name='Sheet1', index=False)
    prd_sample_df.to_excel(writer, sheet_name='prd_sample', index=False)

print("✅ Sheet 'prd_sample' created and updated successfully.")









import pandas as pd

# Step 1: Load the Excel file
file_path = 'your_file.xlsx'

# Read both sheets
sheet1_df = pd.read_excel(file_path, sheet_name='Sheet1')
sheet2_df = pd.read_excel(file_path, sheet_name='Sheet2')

# Step 2: Define column mapping from Sheet1 to Sheet2
column_mapping = {
    'ClaimType': 'ClaimTypeMapped',  # e.g., Sheet1["ClaimType"] → Sheet2["ClaimTypeMapped"]
    'POS': 'POSMapped'
}

# Step 3: Apply transformations before mapping

# Fix ClaimType: Institutional Inpatient → Institutional
sheet1_df['ClaimType'] = sheet1_df['ClaimType'].replace({
    'Institutional Inpatient': 'Institutional',
    'Institutional Outpatient': 'Institutional'  # Optional: group both
})

# Fix POS: if value is numeric 31, make it string and prefix 00 → "0031"
sheet1_df['POS'] = sheet1_df['POS'].apply(lambda x: f"{int(x):04}" if pd.notnull(x) else x)

# Step 4: Copy mapped columns from Sheet1 to Sheet2
for src_col, target_col in column_mapping.items():
    if src_col in sheet1_df.columns and target_col in sheet2_df.columns:
        sheet2_df[target_col] = sheet1_df[src_col]
    else:
        print(f"⚠️ Column not found: {src_col} or {target_col}")

# Step 5: Save the updated Excel with both sheets
with pd.ExcelWriter('updated_output.xlsx', engine='openpyxl', mode='w') as writer:
    sheet1_df.to_excel(writer, sheet_name='Sheet1', index=False)
    sheet2_df.to_excel(writer, sheet_name='Sheet2', index=False)

print("✅ Excel updated with mapped columns and corrected values.")












import pandas as pd

# Step 0: Define relevant claim types
claim_types = ['Institutional Inpatient', 'Institutional Outpatient', 'Professional']
selected_rows = []

# Step 1: Loop through each claim type
for claim_type in claim_types:
    ct_df = merged_df[merged_df['ClaimType'] == claim_type]
    
    # Step 2: Get top 10 diagnosis codes for this claim type
    top_10_diag = ct_df['DiagCode'].value_counts().nlargest(10).index
    
    for diag in top_10_diag:
        diag_subset = ct_df[ct_df['DiagCode'] == diag]
        
        # Top 3 service codes and providers for this diag within claim type
        top_3_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
        top_3_prov = diag_subset['Provider'].value_counts().nlargest(3).index
        
        # Get top 3 rows per service code
        for serv in top_3_serv:
            temp = diag_subset[diag_subset['ServCode'] == serv].head(3)
            selected_rows.append(temp)
        
        # Get top 3 rows per provider
        for prov in top_3_prov:
            temp = diag_subset[diag_subset['Provider'] == prov].head(3)
            selected_rows.append(temp)

    # Step 3: Add top 3 high-dollar claims from this claim type
    top_amount_claims = ct_df.sort_values('AmountPaid', ascending=False).head(3)
    selected_rows.append(top_amount_claims)

# Step 4: Combine everything
final_df = pd.concat(selected_rows).drop_duplicates().reset_index(drop=True)

# Step 5: Export or print
final_df.to_excel('ClaimTypeWise_TopClaims.xlsx', index=False)

print("✅ Final Data Preview:")
print(final_df.head(10))







import pandas as pd

# Step 0: Define valid ClaimTypes and filter
valid_claim_types = ['Institutional Inpatient', 'Institutional Outpatient', 'Professional']
filtered_df = merged_df[merged_df['ClaimType'].isin(valid_claim_types)]

# Step 1: Get top 10 diagnosis codes from filtered data
top_10_diag = filtered_df['DiagCode'].value_counts().nlargest(10).index

# Container for selected rows
selected_rows = []

# Step 2: Loop through each top diagnosis code
for diag in top_10_diag:
    diag_subset = filtered_df[filtered_df['DiagCode'] == diag]
    
    # Top 3 service codes and top 3 providers for this diagnosis
    top_3_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
    top_3_prov = diag_subset['Provider'].value_counts().nlargest(3).index

    # Get top 3 rows for each ServCode
    for serv in top_3_serv:
        temp = diag_subset[diag_subset['ServCode'] == serv].head(3)
        selected_rows.append(temp)
    
    # Get top 3 rows for each Provider
    for prov in top_3_prov:
        temp = diag_subset[diag_subset['Provider'] == prov].head(3)
        selected_rows.append(temp)

# Step 3: Add Top 3 high-dollar claims from each ClaimType
top_amount_claims = (
    filtered_df.sort_values('AmountPaid', ascending=False)
    .groupby('ClaimType')
    .head(3)
)
selected_rows.append(top_amount_claims)

# Step 4: Add Top 3 claims by summary stats (optional: based on highest AmountPaid)
top_stats_claims = (
    filtered_df.groupby('ClaimType', group_keys=False)
    .apply(lambda x: x.sort_values('AmountPaid', ascending=False).head(3))
)
selected_rows.append(top_stats_claims)

# Step 5: Combine all collected rows
final_df = pd.concat(selected_rows).drop_duplicates().reset_index(drop=True)

# Step 6: Export or preview
final_df.to_excel('Filtered_TopClaims_By_Diag_Service_Provider_Pay.xlsx', index=False)

print("✅ Final Data Preview:")
print(final_df.head(10))










import pandas as pd

# Step 1: Get top 10 diagnosis codes
top_10_diag = merged_df['DiagCode'].value_counts().nlargest(10).index

# Final result container
all_selected_rows = []

# Step 2: For each diagnosis code
for diag in top_10_diag:
    diag_subset = merged_df[merged_df['DiagCode'] == diag]
    
    # Top 3 service codes and providers for that diagnosis
    top_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
    top_prov = diag_subset['Provider'].value_counts().nlargest(3).index

    # For each top service code → get top 3 rows
    for serv in top_serv:
        top_rows_serv = diag_subset[diag_subset['ServCode'] == serv].head(3)
        all_selected_rows.append(top_rows_serv)

    # For each top provider → get top 3 rows
    for prov in top_prov:
        top_rows_prov = diag_subset[diag_subset['Provider'] == prov].head(3)
        all_selected_rows.append(top_rows_prov)

# Step 3: Combine and remove duplicates
final_df = pd.concat(all_selected_rows).drop_duplicates().reset_index(drop=True)

# Step 4: Optional — Export to Excel
final_df.to_excel('TopDiag_ServiceProvider_LimitedRows.xlsx', index=False)

# Preview
print(final_df.head())








import pandas as pd

# Step 1: Get Top 10 DiagCodes
top_10_diag = merged_df['DiagCode'].value_counts().nlargest(10).index

# Step 2: Initialize final results list
all_matches = []

# Step 3: Loop through each top diagnosis
for diag in top_10_diag:
    diag_subset = merged_df[merged_df['DiagCode'] == diag]
    
    # Get top 3 service codes and top 3 providers for this diagnosis
    top_3_serv = diag_subset['ServCode'].value_counts().nlargest(3).index
    top_3_prov = diag_subset['Provider'].value_counts().nlargest(3).index
    
    # Filter full rows for either top 3 service codes or top 3 providers
    serv_matches = diag_subset[diag_subset['ServCode'].isin(top_3_serv)]
    prov_matches = diag_subset[diag_subset['Provider'].isin(top_3_prov)]
    
    # Combine and deduplicate
    combined_matches = pd.concat([serv_matches, prov_matches]).drop_duplicates()
    
    # Append to final list
    all_matches.append(combined_matches)

# Step 4: Combine all into a single DataFrame
final_df = pd.concat(all_matches).drop_duplicates().reset_index(drop=True)

# Step 5: Optional — Export to Excel with all fields
final_df.to_excel('TopDiag_FullDetails.xlsx', index=False)

# Preview first few rows
print(final_df.head())












{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2062, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 2089, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 2114, 67, 68, 12357, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 4181, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 2160, 113, 114, 115, 112, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 86152, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 165, 167, 20648, 169, 170, 171, 172, 174, 175, 176, 177, 178, 30899, 179, 181, 182, 183, 184, 185, 187, 188, 190, 191, 193, 2242, 8387, 196, 197, 200, 201, 202, 2251, 204, 207, 41, 211, 212, 213, 215, 217, 2266, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 234, 235, 238, 2287, 240, 241, 242, 4339, 244, 4341, 243, 247, 248, 249, 250, 251, 253, 255, 256, 258, 260, 262, 264, 2315, 267, 269, 4368, 272, 274, 276, 277, 279, 280, 281, 283, 284, 288, 290, 291, 292, 294, 298, 301, 302, 303,
304, 305, 6450, 307, 306, 309, 310, 312, 313, 315, 316, 14653, 320, 10561, 322, 321, 326, 327, 330, 4428, 333, 332, 66, 336, 338, 4436, 342, 344, 347, 348, 349, 69, 4447, 355, 14, 357, 358, 359, 360, 362, 367, 10608, 6515, 375, 378, 381, 384, 2433, 387, 388, 389, 391, 392, 393, 397, 400, 401, 404, 405, 413, 417, 418, 422, 425, 426, 85, 431, 433, 435, 436, 440, 2489, 448, 454, 459, 466, 467, 471, 477, 478, 480, 483, 486, 489, 490, 491, 492, 493, 503, 505, 507, 509, 511, 514, 522, 532, 534, 536, 538, 539, 542, 543, 544, 545, 562, 567, 568, 580, 582, 583, 2632, 588, 6734, 591, 590, 2642, 594, 599, 602, 606, 2662, 621, 4721, 10867, 627, 8821, 648, 651, 653, 654, 655, 664, 671, 672, 680, 682, 683, 136, 687, 688, 689, 692, 693, 694, 697, 710, 715, 21198, 719, 722, 724, 725, 727, 731, 733, 735, 739, 2788, 2789, 2790, 744, 749, 751, 758, 762, 763, 774, 781, 795, 799, 800, 804, 808, 815, 822, 829, 830, 9026, 858, 870, 872, 873, 2935, 894, 895, 896, 2952, 911, 913, 2977, 5030, 954, 957, 960, 965, 25542, 9167, 980, 989, 1003, 7148, 3054, 1018, 1025, 13326, 1041, 1049, 1051, 7199, 1059, 1063, 1077, 1081, 1088, 3157, 1123, 1126, 1136, 5240, 1148, 5246, 3199, 1158, 1160, 1162, 1168, 1172, 7319, 1189, 1200, 1208, 19641, 1213, 3262, 3263, 1215, 1229, 1232, 1238, 1241, 1248, 1252, 1258, 1259, 1265, 1270, 1271, 1274, 1276, 3325, 1280, 1281, 1293, 1300, 7454, 15659, 1325, 1334, 1338, 1355, 1359, 1366, 1367, 1385, 9579, 1414, 1436, 1442, 1463, 1467, 1477, 26066, 13782, 5592, 1506, 1512, 1564, 1584, 30261, 1610, 1612, 3690, 7788, 1654, 5764, 1690, 3744, 11953, 1724, 1729, 1730, 1740, 1751, 1752, 1754, 14058, 1782, 1793, 1802, 1817, 1823, 14135, 1851, 1865, 1874, 1908, 1910, 12156, 1942, 1976, 1983, 6084, 10191, 2021, 2022, 28648, 2029, 2032, 4086)






unique_claims = claim_trend['ClaimType'].unique()
palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Plotly

color_map = {claim: palette[i % len(palette)] for i, claim in enumerate(unique_claims)}









claim_types = claim_trend['ClaimType'].unique()
color_map = {ct: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, ct in enumerate(claim_types)}

# Plot with consistent color mapping
fig = px.bar(
    claim_trend,
    x="YearMonth",
    y="ClaimCount",
    color="ClaimType",
    color_discrete_map=color_map,
    barmode="group",
    title="Monthly Claim Trends"
)








# Ensure your merged_df has the 'YearMonth' column properly formatted
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])
merged_df['YearMonth'] = merged_df['ServDate'].dt.strftime('%Y-%B')

# Step 1: Sort and calculate percentage change (already assumed to be done)
# claim_trend has columns: ['YearMonth', 'ClaimType', 'ClaimCount', 'PercentChange']

# Step 2: Filter for percentage drop below -20%
significant_drop = claim_trend[claim_trend['PercentChange'] < -20]

# Step 3: Get all claims from current and previous month for that ClaimType
all_records = []

for _, row in significant_drop.iterrows():
    curr_ym = row['YearMonth']
    claim_type = row['ClaimType']
    
    # Get index of current month in order list
    idx = month_order.index(curr_ym)
    
    if idx == 0:
        continue  # No previous month to compare
    
    prev_ym = month_order[idx - 1]

    # Filter merged_df for both months and claim type
    relevant_claims = merged_df[
        (merged_df['ClaimType'] == claim_type) &
        (merged_df['YearMonth'].isin([prev_ym, curr_ym]))
    ][['YearMonth', 'ClaimType', 'ClaimID', 'ServCode', 'RevenueCode', 'DiagCode', 'Provider']]
    
    all_records.append(relevant_claims)

# Step 4: Concatenate all results into one DataFrame
final_claims_df = pd.concat(all_records, ignore_index=True)

# Display or export final_claims_df
print(final_claims_df)










import pandas as pd

# Ensure date is in datetime format
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract Year-Month in the format 'YYYY-MMM' (e.g., '2022-Mar')
merged_df['YearMonth'] = merged_df['ServDate'].dt.strftime('%Y-%B')

# Create monthly claim count summary
claim_trend = merged_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='ClaimCount')

# Custom month ordering
month_order = ['2022-March', '2022-July', '2022-August', '2022-September', '2022-October', '2022-November', '2022-December',
               '2023-January', '2023-February', '2023-March', '2023-April', '2023-May', '2023-June', '2023-July', '2023-August', 
               '2023-September', '2024-January', '2024-February', '2024-March', '2024-April', '2024-May', '2024-June', 
               '2024-July', '2024-August', '2024-September', '2025-January', '2025-February', '2025-March', '2025-April', 
               '2025-May', '2025-June']

# Set correct order for plotting/sorting
claim_trend['YearMonth'] = pd.Categorical(claim_trend['YearMonth'], categories=month_order, ordered=True)

# Sort by YearMonth and ClaimType for accurate diff calculation
claim_trend = claim_trend.sort_values(['ClaimType', 'YearMonth'])

# Calculate percentage change
claim_trend['PercentChange'] = claim_trend.groupby('ClaimType')['ClaimCount'].pct_change() * 100

# Optional: Round the percentage change
claim_trend['PercentChange'] = claim_trend['PercentChange'].round(2)

# Display the DataFrame
print(claim_trend.head(15))














import plotly.express as px
import pandas as pd

# Filter for selected claim types
selected_claims = ['Professional', 'Institutional Inpatient', 'Institutional Outpatient']
service_df = merged_df[merged_df['ClaimType'].isin(selected_claims)]

# Group and count
top_service_codes = (
    service_df.groupby(['ClaimType', 'RevCode'])
    .size()
    .reset_index(name="Count")
)

# Ensure RevCode is string
top_service_codes['RevCode'] = top_service_codes['RevCode'].astype(str)

# Get top 1 RevCode per ClaimType
top_service_codes = (
    top_service_codes
    .sort_values('Count', ascending=False)
    .groupby('ClaimType')
    .head(1)
)

# Create figure
fig_find6 = px.bar(
    top_service_codes,
    x='RevCode',
    y='Count',
    color='ClaimType',
    barmode='overlay',
    height=500,
    opacity=0.75,
    text='Count',
    title='Top Revenue Code per Claim Type'
)

# Update trace text
fig_find6.update_traces(textposition='auto')

# Update layout
fig_find6.update_layout(
    xaxis_title='Revenue Code',
    yaxis=dict(title='Count', type='log', showgrid=False, zerolinecolor='#060608'),
    bargap=0.1
)

# 🚨 Identify missing claim types
present_claim_types = top_service_codes['ClaimType'].unique().tolist()
missing_claim_types = [ct for ct in selected_claims if ct not in present_claim_types]

# Add annotation for each missing claim type
for i, missing in enumerate(missing_claim_types):
    fig_find6.add_annotation(
        text=f"No data for {missing}",
        xref="paper", yref="paper",
        x=1.05, y=1 - i * 0.05,  # adjust spacing between notes
        showarrow=False,
        font=dict(color="crimson", size=12)
    )

fig_find6.show()










from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd

# Just to be safe, ensure DOB is in datetime format
merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')

# Define today's date once
today = datetime.today()

# Define a function to calculate age
def calculate_age(dob):
    if pd.isnull(dob):
        return None
    # Convert pandas Timestamp to Python datetime
    dob = dob.to_pydatetime()
    return relativedelta(today, dob).years

# Apply the function to calculate Age
merged_df['Age'] = merged_df['DOB'].apply(calculate_age)

# View result
print(merged_df[['DOB', 'Age']].head())








import pandas as pd
from datetime import datetime

# Assuming your DataFrame is called merged_df and it has a 'DOB' column

# Make sure 'DOB' column is in datetime format
merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')

# Calculate age
today = pd.to_datetime('today')
merged_df['Age'] = (today - merged_df['DOB']).astype('<m8[Y]').astype(int)

# View result
print(merged_df[['DOB', 'Age']].head())









if not common_cols:
    st.error("🚫 No common columns found between Member and Claim data. Cannot proceed with merging.")
else:



import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- Functions ----------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace(" ", "")
    return df

def clean_values(df):
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
    return df

def find_member_differences(df, key_col='MemberID'):
    dupes = df[df.duplicated(subset=key_col, keep=False)]
    differences = []
    for member, group in dupes.groupby(key_col):
        differing_cols = []
        for col in group.columns:
            if group[col].nunique() > 1:
                differing_cols.append(col)
        if differing_cols:
            row = {key_col: member}
            for col in differing_cols:
                row[col] = group[col].unique().tolist()
            differences.append(row)
    return pd.DataFrame(differences)

def generate_claim_report(merged_df):
    claim_count = merged_df['ClaimType'].value_counts()
    claim_percentages = (claim_count / len(merged_df)) * 100
    claim_percent_df = claim_percentages.reset_index()
    claim_percent_df.columns = ['ClaimType', 'Percentage']
    return claim_percent_df

def render_bar_chart(claim_percent_df):
    fig = px.bar(
        claim_percent_df,
        x='ClaimType',
        y='Percentage',
        title='Percentage Distribution of Claim Types',
        text='Percentage',
        color='ClaimType'
    )
    fig.update_traces(
        texttemplate='%{text:.2f}%', 
        textposition='outside',
        marker=dict(line=dict(width=1))
    )
    fig.update_layout(
        bargap=0.1,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        yaxis=dict(title='Percentage', showgrid=False, zerolinecolor='gray')
    )
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("📊 Custom Data Visualization Tool")

with st.sidebar:
    st.header("Upload Excel Files")
    member_file = st.file_uploader("Upload Member File (.xlsx)", type=['xlsx'])
    claim_file = st.file_uploader("Upload Claim File (.xlsx)", type=['xlsx'])

    member_sheet = st.text_input("Enter Member Sheet Name", value="MemberPIC_20250609")
    claim_sheet = st.text_input("Enter Claim Sheet Name", value="Delimited")

    load_data = st.button("Load Data")

if load_data and member_file and claim_file:
    # Read files
    df_Member = pd.read_excel(member_file, sheet_name=member_sheet)
    df_Claim = pd.read_excel(claim_file, sheet_name=claim_sheet)

    st.subheader("🔍 Data Overview")
    st.write("✅ Member Data: ", df_Member.shape)
    st.write(df_Member.columns.tolist())

    st.write("✅ Claim Data: ", df_Claim.shape)
    st.write(df_Claim.columns.tolist())

    # Clean column names
    df_Member = clean_columns(df_Member)
    df_Claim = clean_columns(df_Claim)

    # Multiselect for common columns
    common_cols = list(set(df_Member.columns).intersection(set(df_Claim.columns)))
    selected_cols = st.multiselect("Select Common Column(s) to Merge On", common_cols, default=["MemberID"])

    if selected_cols:
        df_Member = clean_values(df_Member)
        df_Claim = clean_values(df_Claim)

        # Find differences in Member table (optional)
        diff_df = find_member_differences(df_Member, key_col=selected_cols[0])
        if not diff_df.empty:
            st.write("⚠️ Duplicate Member Entries with Differences")
            st.dataframe(diff_df)

        d1_deduped = df_Member.drop_duplicates(subset=selected_cols, keep='first')
        merged_df = pd.merge(df_Claim, d1_deduped, on=selected_cols, how='left')

        st.subheader("🧾 Merged DataFrame Info")
        st.write("Shape:", merged_df.shape)
        st.write("Columns:", merged_df.columns.tolist())
        st.dataframe(merged_df.head())

        if st.button("📈 Generate Claim Type Distribution Report"):
            try:
                claim_percent_df = generate_claim_report(merged_df)
                st.dataframe(claim_percent_df)
                st.plotly_chart(render_bar_chart(claim_percent_df), use_container_width=True)
            except Exception as e:
                st.error(f"Error generating chart: {e}")













_________________________
def find_column_in_dfs(dfs, target_column):
    found_in = []

    for idx, df in enumerate(dfs):
        if target_column in df.columns:
            found_in.append((idx, df.columns.tolist()))

    if found_in:
        print(f"\n✅ Column '{target_column}' found in {len(found_in)} DataFrame(s):\n")
        for idx, columns in found_in:
            print(f"➡️  DataFrame #{idx} contains it. Columns: {columns}")
    else:
        print(f"\n❌ Column '{target_column}' not found in any DataFrame.")

    return found_in











import os
import re
import pandas as pd

def collect_usecase_sheets(root_folder):
    dfs = []
    count = 0
    ARCHIVE = 'ARCHIVE'  # Define the ARCHIVE keyword or folder name

    # Regex pattern to match all variants of "Use Cases" sheet names
    use_case_pattern = re.compile(
        r'(?:\d+(?:\.\d+)?\s*\.*\s*)?(?:SKN_)?(?:Non-)?Aliessa_?Use_?Cases|Use[\s_]?Cases(?:_[A-Za-z0-9]+)?|Use_?Cases[\sA-Za-z0-9_]*'
    )

    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.xlsm') and file.startswith('OMSAP'):
                file_path = os.path.join(dirpath, file)

                if 'ARCHIVE' in file_path.upper() or 'MAILS' in file_path.upper():
                    continue

                count += 1
                try:
                    xls = pd.ExcelFile(file_path)
                    sheet_names = xls.sheet_names

                    matched_sheet = next(
                        (s for s in sheet_names if use_case_pattern.fullmatch(s.strip())), None
                    )

                    if matched_sheet:
                        df = pd.read_excel(file_path, sheet_name=matched_sheet)
                        dfs.append(df)
                    else:
                        folder = file_path.split(os.sep)[2] if len(file_path.split(os.sep)) > 2 else "Unknown"
                        print(f"No 'Use Cases' sheet found in: {file_path} (Folder: {folder})")

                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

    print(f"Total matching files: {count}")
    return dfs












(?:\d+(?:\.\d+)?\s*\.*\s*)?(?:SKN_)?(?:Non-)?Aliessa_?Use_?Cases|Use[\s_]?Cases(?:_[A-Za-z0-9]+)?|Use_?Cases[\sA-Za-z0-9_]*











B20 (HIV Disease)
• Highest diagnosis consistently across all months
• Peaked in Oct 2024, slight drop by Mar 2025

E11.9 (Type 2 Diabetes without complications)
• Steady increase till Jan 2025
• Sharp drop observed in Mar 2025


F11.20 (Opioid dependence, uncomplicated)
• Rose steadily till Oct 2024
• Declined significantly after Feb 2025

I10 (Essential Hypertension)
• Gradual upward trend throughout
• Minor dip noted after Feb 2025

N18.6 (End stage renal disease)
• Very stable over the entire period
• Minimal fluctuations month-to-month

R68.89 (Other general symptoms and signs)
• Growth till Oct 2024, then leveled off
• Slight drop by Mar 2025

R69 (Unknown causes of morbidity/mortality)
• Minor rise around Dec 2024
• Dropped again by Mar 2025

Z00.00 (General medical exam without abnormal findings)
• Spike seen during Oct–Dec 2024
• Sharp decline by Mar 2025

Z21 (Asymptomatic HIV infection status)
• Fairly stable till Jan 2025
• Sudden decrease in the final month

Z76.89 (Other specified encounters)
• Consistently low volume throughout
• Nearly flat by Mar 2025


import pandas as pd
import plotly.express as px

# Step 1: Ensure 'ServDate' is datetime
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'], errors='coerce')

# Step 2: Create Year-Month column
merged_df['YearMonth'] = merged_df['ServDate'].dt.to_period('M').astype(str)

# Step 3: Get top 10 diagnosis codes
top_10_diags = (
    merged_df['Diag']
    .value_counts()
    .head(10)
    .index
)

# Step 4: Filter for only top 10
diag_trend_df = merged_df[merged_df['Diag'].isin(top_10_diags)]

# Step 5: Group by YearMonth and Diag
trend_counts = (
    diag_trend_df.groupby(['YearMonth', 'Diag'])
    .size()
    .reset_index(name='Count')
)

# Step 6: Sort YearMonth properly
trend_counts['YearMonth'] = pd.to_datetime(trend_counts['YearMonth'])
trend_counts = trend_counts.sort_values('YearMonth')

# Step 7: Plot
fig = px.line(
    trend_counts,
    x='YearMonth',
    y='Count',
    color='Diag',
    markers=True,
    title='Trend Analysis of Top 10 Diagnosis Codes',
)

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Claim Count',
    xaxis=dict(
        tickformat="%b %Y",  # Format like 'Jan 2024'
        dtick="M1"
    ),
    plot_bgcolor='white',
    font_color="#303030",
    legend_title='Diagnosis Code'
)

fig.show()










1,,,,,

import pandas as pd
import plotly.express as px

# Replace this with your actual data
merged_df = pd.read_csv("your_data.csv")

# Top 5 providers by total bill
top_billers = (
    merged_df.groupby('Provider')['BillAmount']
    .sum()
    .reset_index()
    .sort_values(by='BillAmount', ascending=False)
    .head(5)
)

fig1 = px.bar(
    top_billers,
    x='Provider',
    y='BillAmount',
    text='BillAmount',
    title='Top 5 Providers by Total Billed Amount',
    color_discrete_sequence=['#a0d8ef']  # light blue
)
fig1.update_traces(textposition='outside')
fig1.update_layout(yaxis_title='Total Bill Amount', plot_bgcolor='white')
fig1.show()
2,,,,,,
top_submitters = (
    merged_df.groupby('Provider')
    .size()
    .reset_index(name='ClaimCount')
    .sort_values(by='ClaimCount', ascending=False)
    .head(5)
)

fig2 = px.bar(
    top_submitters,
    x='Provider',
    y='ClaimCount',
    text='ClaimCount',
    title='Top 5 Providers by Number of Claims Submitted',
    color_discrete_sequence=['#c6e2ff']  # soft blue
)
fig2.update_traces(textposition='outside')
fig2.update_layout(yaxis_title='Claim Count', plot_bgcolor='white')
fig2.show()
3,,,,,,

denied_df = merged_df[merged_df['PayStat'] == 'D']

top_denied = (
    denied_df.groupby('Provider')
    .size()
    .reset_index(name='DeniedCount')
    .sort_values(by='DeniedCount', ascending=False)
    .head(5)
)

fig3 = px.bar(
    top_denied,
    x='Provider',
    y='DeniedCount',
    text='DeniedCount',
    title='Top 5 Providers by Number of Denied Claims',
    color_discrete_sequence=['#ffcccc']  # light red
)
fig3.update_traces(textposition='outside')
fig3.update_layout(yaxis_title='Denied Claims', plot_bgcolor='white')
fig3.show()









import pandas as pd
import plotly.express as px

# Convert 'ServDate' to datetime if it's not already
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract Year-Month in desired format (e.g., Jun-2024)
merged_df['YearMonth'] = merged_df['ServDate'].dt.strftime('%b-%Y')

# Ensure proper month order for plotting
month_order = ['Jun-2024', 'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024',
               'Nov-2024', 'Dec-2024', 'Jan-2025', 'Feb-2025', 'Mar-2025']
merged_df['YearMonth'] = pd.Categorical(merged_df['YearMonth'], categories=month_order, ordered=True)

# Filter only denied claims
denied_df = merged_df[merged_df['PayStat'] == 'D']

# Group by month and claim type to get denial counts
denial_trend = denied_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='DenialCount')

# Plot line chart
fig = px.line(
    denial_trend,
    x='YearMonth',
    y='DenialCount',
    color='ClaimType',
    markers=True,
    title='Monthly Denial Trend by Claim Type'
)

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Denials',
    font_color="#303030",
    plot_bgcolor='white',
    legend_title_text='Claim Type'
)

fig.update_traces(line=dict(width=3), marker=dict(size=6))

fig.show()








import pandas as pd
import plotly.express as px

# Assume 'merged_df' is your dataframe and has columns 'ServDate' and 'ClaimType'
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract 'Year-Month' format from service date
merged_df['YearMonth'] = merged_df['ServDate'].dt.to_period('M').astype(str)

# Group by YearMonth and ClaimType to get the count of claims per month
claim_trend = merged_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='ClaimCount')

# Create ordered list of expected months (June 2024 to March 2025)
month_order = ['2024-06', '2024-07', '2024-08', '2024-09', '2024-10',
               '2024-11', '2024-12', '2025-01', '2025-02', '2025-03']

# Ensure all months show in the plot, even if counts are 0
claim_trend['YearMonth'] = pd.Categorical(claim_trend['YearMonth'], categories=month_order, ordered=True)
claim_trend = claim_trend.sort_values('YearMonth')

# Plot line graph
fig = px.line(
    claim_trend,
    x='YearMonth',
    y='ClaimCount',
    color='ClaimType',
    markers=True,
    line_shape='spline',
    title='Monthly Trend of Each Claim Type (Jun 2024 - Mar 2025)',
    color_discrete_sequence=px.colors.qualitative.Safe
)

fig.update_traces(line=dict(width=2))

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Claims',
    font_color="#303030",
    height=500,
    template='plotly_white',
    legend_title_text='Claim Type',
    xaxis=dict(
        tickangle=45,
        type='category',
        categoryorder='array',
        categoryarray=month_order
    )
)

fig.show()









import pandas as pd
import plotly.express as px

# Filter for relevant claim types
claim_status_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional Inpatient', 'Institutional Outpatient'])]

# Map PayStat codes to readable labels
claim_status_df['PayStat'] = claim_status_df['PayStat'].map({'P': 'Paid', 'D': 'Denied'})

# Drop rows with missing PayStat after mapping
claim_status_df = claim_status_df.dropna(subset=['PayStat'])

# Group by ClaimType and PayStat
claim_status_count = claim_status_df.groupby(['ClaimType', 'PayStat']).size().reset_index(name='Count')

# Calculate percentage
total_counts = claim_status_count.groupby('ClaimType')['Count'].transform('sum')
claim_status_count['Percentage'] = (claim_status_count['Count'] / total_counts * 100).round(2)

# Label for display
claim_status_count['label'] = claim_status_count['Count'].astype(str) + ' (' + claim_status_count['Percentage'].astype(str) + '%)'

# Plot grouped bar chart
fig_find4 = px.bar(
    claim_status_count,
    x='ClaimType',
    y='Count',
    color='PayStat',
    barmode='group',
    text='label',
    title='Paid vs Denied Claim Counts by Claim Type',
    height=500,
    color_discrete_map={'Paid': '#90ee90', 'Denied': '#ff9999'}
)

# Style update
fig_find4.update_traces(textposition='outside')
fig_find4.update_layout(
    font_color="#303030",
    xaxis_title='Claim Type',
    yaxis=dict(title='Claim Count', type='log', showgrid=False, zerolinecolor='#DBDBDB'),
    legend_title='Claim Status',
    bargap=0.3
)







import pandas as pd
import plotly.express as px

# Convert ServDate to datetime
merged_df['ServDate'] = pd.to_datetime(merged_df['ServDate'])

# Extract year-month
merged_df['YearMonth'] = merged_df['ServDate'].dt.to_period('M').astype(str)

# Group by YearMonth and ClaimType
claim_trend = merged_df.groupby(['YearMonth', 'ClaimType']).size().reset_index(name='ClaimCount')

# Sort by YearMonth for proper line flow
claim_trend = claim_trend.sort_values('YearMonth')

# Line plot for each claim type
fig = px.line(
    claim_trend,
    x='YearMonth',
    y='ClaimCount',
    color='ClaimType',
    markers=True,
    line_shape='spline',
    title='Monthly Trend of Each Claim Type (Jun 2024 - Mar 2025)',
    color_discrete_sequence=px.colors.qualitative.Safe  # light + distinguishable
)

# Update layout
fig.update_traces(line=dict(width=2))
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Claims',
    font_color="#303030",
    height=500,
    template='plotly_white',
    legend_title_text='Claim Type',
    xaxis=dict(tickangle=45)
)

fig.show()












import plotly.express as px

# Bar chart with thicker bars
fig = px.bar(claim_percent_df,
             x='ClaimType',
             y='Percentage',
             title='Percentage Distribution of Claim Types',
             text='Percentage',
             color='ClaimType')

fig.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside',
    marker=dict(line=dict(width=0)),  # No border, solid bars
)

fig.update_layout(
    width=800,
    height=500,
    bargap=0.1,  # Reduce space between bars
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    yaxis_title='Percentage'
)

fig.show()









Hi Ishita, Supreeth,

I've added two more findings to the deck. As requested, I’ve included interactive graphs by converting them into HTML outputs and linking them to the graph tiles in the presentation. Please download the ZIP file, open the presentation in slideshow mode, and click on any graph tile to view the corresponding interactive graph.

Thanks!







claims with bill amount over 10K, there are no professional claims; only institutional claims are present, with B20 being the most common diagnosis code in outpatient (49,955 claims), followed by Z21, Z00.00, R69, and F11.20.





Institutional - Outpatient has T1019 as the most used service code with 33,840 claims.

Institutional - Inpatient sees 99232 as the top service code with 1,679 claims.

Professional has 99309 as the most frequent service code with 105 claims.










import pandas as pd
import plotly.express as px

# Categorize Age Function
def categorize_age(age):
    if age == 0:
        return 'New Born'
    elif age <= 18:
        return '1 to 18 years'
    elif age <= 35:
        return '19 to 35 years'
    elif age <= 59:
        return '36 to 59 years'
    else:
        return '60 and over'

# Apply age category
plot_df['AgeGroup'] = plot_df['Age'].apply(categorize_age)

# Group by AgeGroup and ClaimType
plot_df = plot_df.groupby(['AgeGroup', 'ClaimType'])['ClaimType'].count().reset_index(name='ClaimTypeCount')

# Order age groups
age_order = ['New Born', '1 to 18 years', '19 to 35 years', '36 to 59 years', '60 and over']
plot_df['AgeGroup'] = pd.Categorical(plot_df['AgeGroup'], categories=age_order, ordered=True)
plot_df = plot_df.sort_values(['AgeGroup', 'ClaimType'])

# Create bar chart
fig_find1 = px.bar(
    plot_df,
    x='AgeGroup',
    y='ClaimTypeCount',
    color='ClaimType',
    text='ClaimTypeCount',
    opacity=0.75,
    barmode='group',
    title='Total Number of Claims by Age Group and Claim Type'
)

# Update chart aesthetics
fig_find1.update_traces(
    texttemplate='%{text:.0f}',
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig_find1.update_layout(
    font_color="#383038",
    bargroupgap=0.05,
    bargap=0.2,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
    xaxis=dict(title='Age Group', showgrid=False),
    yaxis=dict(
        title='Claim Count',
        showgrid=False,
        zerolinecolor="#DBDBDB",
        showline=True,
        linecolor="#080808",
        linewidth=2,
        type='log'
    )
)
















fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Total Number of Claims by Age Group and Claim Type",
        "Top 5 Frequent Diagnoses by Age Group",
        "Top Claim Type by Number of Providers",
        "Paid vs Denied Claim Counts by Claim Type"
    ),
    specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "pie"}, {"type": "bar"}]]
)

# Add each figure to subplot
fig_dashboard.add_trace(fig_find1, row=1, col=1)
fig_dashboard.add_trace(fig_find1b, row=1, col=1)

fig_dashboard.add_trace(fig_find2, row=1, col=2)
fig_dashboard.add_trace(fig_find2b, row=1, col=2)























import pandas as pd
import plotly.express as px

# Step 1: Filter Institutional and Professional
claim_df = merged_df[merged_df['ClaimType'].isin(['Institutional', 'Professional'])]

# Step 2: Count total claims per ClaimType
claim_counts = claim_df.groupby('ClaimType').size().reset_index(name='TotalCount')

# Step 3: Filter only those ClaimTypes with count > 10,000
eligible_types = claim_counts[claim_counts['TotalCount'] > 10000]['ClaimType']
filtered_df = claim_df[claim_df['ClaimType'].isin(eligible_types)]

# Step 4: Count DiagnosisCode occurrences
top_diag = (
    filtered_df.groupby(['ClaimType', 'DiagnosisCode'])
    .size()
    .reset_index(name='Count')
    .sort_values(['ClaimType', 'Count'], ascending=[True, False])
    .groupby('ClaimType')
    .head(5)
)

# Step 5: Visualization
fig = px.bar(
    top_diag,
    x='DiagnosisCode',
    y='Count',
    color='ClaimType',
    barmode='group',
    text='Count',
    title='Top 5 Diagnosis Codes by Claim Type (Count > 10K)'
)

fig.update_traces(
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    xaxis_title='Diagnosis Code',
    yaxis_title='Claim Count',
    xaxis_tickangle=45,
    height=500
)

fig.show()








Institutional – Outpatient claims account for the majority of provider activity, representing 70.5% of all claim submissions. This indicates a strong emphasis on outpatient care across the provider network, likely driven by cost efficiency, accessibility, and broader coverage models.

 denial and payment rates are approximately 57.1% denied and 42.9% paid for Institutional Inpatient, 19.4% denied and 80.6% paid for Institutional Outpatient, and 22.8% denied and 77.2% paid for Professional services.





Diagnosis code B20 (HIV disease) consistently ranks as the top diagnosis for patients aged 19 and above, with the highest concentration seen in the 36–59 age group




“With Institutional claims comprising 86.16%—dominated by top provider JACOBI MEDICAL CENTER and top service code 99232—and Professional claims at just 0.123%—led by IVEY GRACE and service code 99309—the data reveals a sharp concentration of volume and impact around a few key players.”








import plotly.express as px

# Filter for selected claim types
benefit_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional Inpatient'])]

# Group and count
benefit_dist = benefit_df.groupby(['ClaimType', 'BenPkg']).size().reset_index(name='Count')

# Top 5 benefit codes per claim type
benefit_dist = benefit_dist.sort_values('Count', ascending=False).groupby('ClaimType').head(5)

# Bar chart
fig5 = px.bar(
    benefit_dist,
    x='BenPkg',
    y='Count',
    color='ClaimType',
    barmode='group',
    text='Count',
    title='Top 5 Benefit Codes by Claim Type',
    height=500
)

fig5.update_traces(textposition='outside')
fig5.update_layout(
    xaxis_title='Benefit Code',
    yaxis_title='Count',
    bargap=0.3,
    font_color="#303030",
    legend_title='Claim Type'
)

fig5.show()










import plotly.express as px

# Filter for relevant claim types
claim_status_df = merged_df[merged_df['ClaimType'].isin([
    'Professional', 'Institutional Inpatient', 'Institutional Outpatient'
])]

# Map PayStat codes
claim_status_df['PayStat'] = claim_status_df['PayStat'].map({
    'P': 'Paid',
    'D': 'Denied'
})

# Drop rows with NaN PayStat after mapping
claim_status_df = claim_status_df.dropna(subset=['PayStat'])

# Group and count
claim_status_count = claim_status_df.groupby(['ClaimType', 'PayStat']).size().reset_index(name='Count')

# Plot as grouped bar
fig = px.bar(
    claim_status_count,
    x='ClaimType',
    y='Count',
    color='PayStat',
    barmode='group',
    text='Count',
    title='Paid vs Denied Claim Counts by Claim Type',
    height=500
)

fig.update_traces(textposition='outside')
fig.update_layout(
    font_color="#303030",
    xaxis_title='Claim Type',
    yaxis_title='Claim Count',
    legend_title='Claim Status',
    bargap=0.3
)

fig.show()






import plotly.express as px

# Filter relevant claim types
claim_status_df = merged_df[merged_df['ClaimType'].isin([
    'Professional', 'Institutional Inpatient', 'Institutional Outpatient'
])]

# Map PayStat codes
claim_status_df['PayStat'] = claim_status_df['PayStat'].map({
    'P': 'Paid',
    'D': 'Denied'
})

# Drop rows with NaN PayStat after mapping
claim_status_df = claim_status_df.dropna(subset=['PayStat'])

# Group and count
claim_status_pie = claim_status_df.groupby(['ClaimType', 'PayStat']).size().reset_index(name='Count')

# Sunburst chart
fig2 = px.sunburst(
    claim_status_pie,
    path=['ClaimType', 'PayStat'],
    values='Count',
    title='Percentage of Paid vs Denied Claims by Claim Type'
)

fig2.update_traces(textinfo="label+percent entry")

fig2.show()







1 

import plotly.express as px
import pandas as pd

# Count of providers per claim type
provider_claim_counts = merged_df.groupby(['ClaimType'])['Provider'].nunique().reset_index(name='ProviderCount')

# Get top 1
top_provider_claim = provider_claim_counts.sort_values('ProviderCount', ascending=False).head(1)

fig1 = px.pie(top_provider_claim, names='ClaimType', values='ProviderCount',
              title='Top Claim Type by Number of Providers', hole=0.3)

fig1.show()


2

claim_status_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

claim_status_pie = claim_status_df.groupby(['ClaimType', 'ClaimStatus']).size().reset_index(name='Count')

fig2 = px.sunburst(claim_status_pie, path=['ClaimType', 'ClaimStatus'], values='Count',
                   title='Denied vs Paid Claim Distribution - Professional & Institutional')

fig2.show()

3

service_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

top_service_codes = service_df.groupby(['ClaimType', 'ServiceCode']).size().reset_index(name='Count')
top_service_codes = top_service_codes.sort_values('Count', ascending=False).groupby('ClaimType').head(10)

fig3 = px.bar(top_service_codes, x='ServiceCode', y='Count', color='ClaimType',
              barmode='group', text='Count', title='Top Service Codes by Claim Type')

fig3.update_traces(textposition='outside')
fig3.update_layout(xaxis_title='Service Code', yaxis_title='Count')

fig3.show()

4
diagnosis_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

top_diag_codes = diagnosis_df.groupby(['ClaimType', 'DiagnosisCode']).size().reset_index(name='Count')
top_diag_codes = top_diag_codes.sort_values('Count', ascending=False).groupby('ClaimType').head(10)

fig4 = px.bar(top_diag_codes, x='DiagnosisCode', y='Count', color='ClaimType',
              barmode='group', text='Count', title='Top Diagnosis Codes by Claim Type')

fig4.update_traces(textposition='outside')
fig4.update_layout(xaxis_title='Diagnosis Code', yaxis_title='Count')

fig4.show()

5


benefit_df = merged_df[merged_df['ClaimType'].isin(['Professional', 'Institutional'])]

benefit_dist = benefit_df.groupby(['ClaimType', 'BenefitCode']).size().reset_index(name='Count')
benefit_dist = benefit_dist.sort_values('Count', ascending=False).groupby('ClaimType').head(5)

fig5 = px.sunburst(benefit_dist, path=['ClaimType', 'BenefitCode'], values='Count',
                   title='Benefit Code Distribution Across Claim Types')

fig5.show()








import pandas as pd
import plotly.express as px
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)
px.defaults.template = "plotly_white"

# Create a copy
plot_df = merged_df.copy()

# Categorize Age groups
def categorize_age(age):
    if age == 0:
        return 'New Born'
    elif age <= 18:
        return '1 to 18 years'
    elif age <= 35:
        return '19 to 35 years'
    elif age <= 59:
        return '36 to 59 years'
    else:
        return '60 and over'

plot_df["AgeGroup"] = plot_df['Age'].apply(categorize_age)

# Group by AgeGroup and ClaimType and count
plot_df = plot_df.groupby(['AgeGroup', 'ClaimType'])['ClaimType'].count().reset_index(name='ClaimTypeCount')

# Sort by AgeGroup for better visual order
age_order = ['New Born', '1 to 18 years', '19 to 35 years', '36 to 59 years', '60 and over']
plot_df['AgeGroup'] = pd.Categorical(plot_df['AgeGroup'], categories=age_order, ordered=True)
plot_df = plot_df.sort_values(['AgeGroup', 'ClaimType'])

# Optional: Save to Excel
# plot_df.to_excel("Average_Bill_Amount_by_Age_and_ClaimType.xlsx", index=False)

# Create bar chart
fig = px.bar(
    plot_df,
    x='AgeGroup',
    y='ClaimTypeCount',
    color='ClaimType',
    text='ClaimTypeCount',
    opacity=0.75,
    barmode='group',
    height=500,
    title="Total Number of Claims by Age Group and Claim Type"
)

# Update styling
fig.update_traces(
    texttemplate='%{text:.0f}',
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    bargroupgap=0.05,
    bargap=0.3,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    xaxis=dict(title='Age Group', showgrid=False),
    yaxis=dict(title='Claim Count', showgrid=False, zerolinecolor='#DBDBDB', showline=True, linecolor='#DBDBDB', linewidth=2)
)

fig.show()









import xml.etree.ElementTree as ET
from collections import defaultdict
import re

def flatten_xml(elem, result, tag_order, prefix=""):
    for child in elem:
        tag = child.tag.split("}")[-1]
        key = f"{prefix}{tag}".lower()
        if key not in tag_order:
            tag_order.append(key)
        if child.text and child.text.strip():
            result[key].append(child.text.strip())
        flatten_xml(child, result, tag_order, key + "_")

def parse_multiple_xml_etree(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to match all <ConvertedSupplierInvoice>...</ConvertedSupplierInvoice> blocks
    pattern = re.compile(r"<ConvertedSupplierInvoice\b.*?>.*?</ConvertedSupplierInvoice>", re.DOTALL)
    matches = pattern.findall(content)

    all_rows = []

    for xml_fragment in matches:
        try:
            # Ensure it's valid XML before processing
            elem = ET.fromstring(xml_fragment)
            result = defaultdict(list)
            tag_order = []
            flatten_xml(elem, result, tag_order)

            max_len = max((len(v) for v in result.values()), default=0)
            for k in result:
                while len(result[k]) < max_len:
                    result[k].append("")

            all_rows.append((result, tag_order.copy()))
        except ET.ParseError as e:
            print(f"Skipped malformed XML block: {e}")
            continue

    return all_rows








# scrap

# Strip currency symbols, commas, etc.
df['BillAmt'] = df['BillAmt'].astype(str).str.replace(r'[^\d.]', '', regex=True)

# Convert to numeric
df['BillAmt'] = pd.to_numeric(df['BillAmt'], errors='coerce')

# Drop rows where BillAmt couldn't be converted
df = df.dropna(subset=['BillAmt'])






import pandas as pd
import plotly.express as px

# Bin BillAmt into ranges
bins = [475, 5000, 20000, 50000, 100000, 250000, 500000, 813439]
labels = ['<5K', '5K–20K', '20K–50K', '50K–100K', '100K–250K', '250K–500K', '>500K']
df['BillAmtBin'] = pd.cut(df['BillAmt'], bins=bins, labels=labels, include_lowest=True)

# Create AgeGroup
df['AgeGroup'] = df['Age'].apply(
    lambda x: '18-29' if x < 30 else
              '30-44' if x < 45 else
              '45-59' if x < 60 else
              '60+'
)

# Group the data to get counts
grouped_df = df.groupby(['BillAmtBin', 'ClaimType', 'Race', 'AgeGroup']).size().reset_index(name='Count')

# Create bar plot
fig = px.bar(
    grouped_df,
    x='BillAmtBin',
    y='Count',
    color='ClaimType',
    facet_col='Race',
    facet_row='AgeGroup',
    barmode='group',
    text='Count',
    title='Claim Count by Bill Amount Bin, Race, AgeGroup and Claim Type'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    font_color="#303030",
    xaxis_title="Bill Amount Range",
    yaxis_title="Number of Claims",
    height=900,
    bargap=0.3
)

fig.show()
----------------------------------------------------------------

# Group and count based on all requested dimensions
grouped_df = df.groupby(
    ['BillAmtBin', 'ClaimType', 'Race', 'AgeGroup', 'GrpNum', 'BenefitPackage']
).size().reset_index(name='Count')

# Create bar chart with faceting on Race, AgeGroup, GrpNum, BenefitPackage
fig = px.bar(
    grouped_df,
    x='BillAmtBin',
    y='Count',
    color='ClaimType',
    facet_col='BenefitPackage',   # Can also try 'Race' here depending on focus
    facet_row='GrpNum',           # Can also try 'AgeGroup' or 'Race' here
    barmode='group',
    text='Count',
    title='Claim Count by Bill Amount Bin, Claim Type, Benefit Package, and GrpNum'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    font_color="#303030",
    xaxis_title="Bill Amount Range",
    yaxis_title="Number of Claims",
    height=1200,
    bargap=0.3
)

fig.show()














Females lack claims for Group Home and Residential Substance Abuse Treatment, both of which are present for males (82 and 15 claims respectively).

Males have no claims for Mobile Unit services, which have 12 claims under females, indicating possible access or utilization differences.

Other claim types dominate for both genders, but especially for females (116,679), far exceeding all other categories, suggesting broad categorization or unclassified services.

Institutional - Outpatient claims are significantly higher in males (158,542) compared to females (71,160), indicating possible gender-based differences in outpatient service utilization.

Telehealth claims are more frequent in females (418) than males (257), reflecting either greater access, acceptance, or necessity among females for remote care.















Step 1: Removing Duplicate Members and Merging Datasets
To remove duplicate entries of members based on MemberNum in df1 while keeping the first occurrence.
To identify common columns between the cleaned df1 and df2 for potential merging.
Note – Validation of Member-Level Data Consistency
I checked the data with Member num i can see the duplicate values only in the Ethnicity. we can confirm that data is duplicated If my understanding is wrong I will make that change accordingly.

Profiling of Categorical Columns
To better understand the structure and distribution of non-numeric (categorical) data in the merged_d
# Step 1: Get the description of categorical columns
cat_profile = merged_df.select_dtypes(include=['object']).describe()

# Step 2: Store in Excel with sheet name
with pd.ExcelWriter('data_profiling.xlsx', engine='openpyxl') as writer:
    cat_profile.to_excel(writer, sheet_name='Categorical Profiling')


# Group by MemberNum and get number of unique provider names
member_provider_counts = df1.groupby('MemberNum')['ProvName'].nunique()

# Count members who visited more than one provider
multi_provider_members_count = (member_provider_counts > 1).sum()

print(f"Number of members who visited more than one provider: {multi_provider_members_count}")

multi_provider_members = member_provider_counts[member_provider_counts > 1].index.tolist()



The total number of diagnoses increases with age, with the highest counts in the "60 and over" and "45 to 59 years" groups.

One diagnosis category dominates across all age groups, especially in older ones (blue bar segment).

The "18 to 29 years" group has the lowest total diagnosis count, with very small bar segments.

Diagnosis diversity is greater in older age groups, indicated by more and larger colored segments in their bars.





Demographic & Provider Profiling: Analyzed key categorical features (Age, MemberNum, and ProvName) to understand claim distribution and average billing trends across member groups and providers.

Claim Concentration: Identified which age groups and providers had the highest number of claimholders, helping to spot clusters of utilization.

Cost Insights: Computed the average bill amount for each category, providing a view into which segments (e.g., certain providers or age groups) tend to incur higher costs




Institutional - Inpatient claims consistently show the highest average bill amounts across all age groups, with the 18 to 29 years group peaking at $4,239
Telehealth and End-Stage Renal Disease Treatment also show relatively high average bill amounts in older age groups (particularly in the 60 and over and 45 to 59 years categories)
Young adults (18 to 29 years) incur lower costs in most claim types, except for the unusually high Institutional - Inpatient expenses.
Services like Independent Lab, Mobile Unit, Other, and Group Home show consistently lower billing amounts.

ClaimTypes such as Mobile Unit, Group Home, and Residential Substance Abuse Treatment are absent in one or more age groups, suggesting:
These services are either not utilized or not recorded for those specific age ranges.
For example, Group Home and Mobile Unit are not present in the 30 to 44 and 45 to 59 age groups.
Telehealth, End-Stage Renal Disease Treatment, and Ambulance - Land tend to have higher average bills for older groups like 60 and over











from plotly.offline import plot, iplot, init_notebook_mode
import plotly.express as px

init_notebook_mode(connected=True)
px.defaults.template = "plotly_white"

# Copy and create age bins
plot_df = merged_df.copy()
plot_df["AgeGroup"] = plot_df['Age'].apply(
    lambda i: '18 to 29 years' if i < 30 else
              '30 to 44 years' if i < 45 else
              '45 to 59 years' if i < 60 else
              '60 and over'
)

# Count number of diagnoses per AgeGroup
diag_counts = (
    plot_df.groupby(['AgeGroup', 'Diag'])['MemberNum']
    .count()
    .reset_index(name='DiagCount')
)

# Get most frequent diagnosis per AgeGroup
top_diag_per_age = diag_counts.sort_values('DiagCount', ascending=False).groupby('AgeGroup').head(1)

# Plot
fig = px.bar(
    top_diag_per_age,
    x='AgeGroup',
    y='DiagCount',
    color='Diag',
    height=500,
    text='DiagCount',
    title="Most Frequent Diagnosis by Age Group"
)

fig.update_traces(
    texttemplate="%{text:,.0f}",
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    bargap=0.3,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    xaxis=dict(title='Age Group', showgrid=False),
    yaxis=dict(
        title='Number of Diagnoses',
        showgrid=False,
        zerolinecolor='#DBDBDB',
        showline=True,
        linecolor='#DBDBDB',
        linewidth=2
    )
)

fig.show()




-----------------------------------------------------

import plotly.express as px

# Group the data by Gender and ClaimType, and count frequencies
claim_counts = (
    df.groupby(['Gender', 'ClaimType'])['MemberNum']
    .count()
    .reset_index(name='ClaimCount')
)

# Plot with Plotly
fig = px.bar(
    claim_counts,
    x='Gender',
    y='ClaimCount',
    color='ClaimType',
    barmode='group',
    text='ClaimCount',
    title="Frequency of ClaimType by Gender"
)

# Styling the chart
fig.update_traces(
    texttemplate="%{text:,}",
    textposition='outside',
    marker_line=dict(width=1, color='#303030')
)

fig.update_layout(
    font_color="#303030",
    bargap=0.3,
    legend_title_text='Claim Type',
    xaxis=dict(title='Gender', showgrid=False),
    yaxis=dict(title='Number of Claims', showgrid=False, zerolinecolor='#DBDBDB')
)

fig.show()

