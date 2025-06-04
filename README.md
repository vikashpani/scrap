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

