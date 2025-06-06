
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

