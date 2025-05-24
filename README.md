# scrap
Step 1: Removing Duplicate Members and Merging Datasets
To remove duplicate entries of members based on MemberNum in df1 while keeping the first occurrence.
To identify common columns between the cleaned df1 and df2 for potential merging.
Note – Validation of Member-Level Data Consistency
I checked the data with Member num i can see the duplicate values only in the Ethnicity. we can confirm that data is duplicated If my understanding is wrong I will make that change accordingly.

# Group by MemberNum and get number of unique provider names
member_provider_counts = df1.groupby('MemberNum')['ProvName'].nunique()

# Count members who visited more than one provider
multi_provider_members_count = (member_provider_counts > 1).sum()

print(f"Number of members who visited more than one provider: {multi_provider_members_count}")

multi_provider_members = member_provider_counts[member_provider_counts > 1].index.tolist()





















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
