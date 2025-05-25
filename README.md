# scrap
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
