# Sensegras_-Assignment

Insights from the Data

1. Top 5 countries with the most wine reviews <br />
US          48153   <br />
France      19932   <br />
Italy       11370   <br />
Portugal     4038   <br />
Chile        3630   <br />
Code:<br />
top_countries = df['country'].value_counts().head(5)<br />
print("Top 5 countries with the most wine reviews:")<br />
print(top_countries)<br />

2. Most common wine varieties<br />
Pinot Noir                             13272<br />
Chardonnay                             11753<br />
Cabernet Sauvignon                     9472<br />
Red Blend                              8946<br />
Bordeaux-style Red Blend               6915<br />
Code:<br />
common_varieties = df['variety'].value_counts().head(5)<br />
print("\nMost common wine varieties:")<br />
print(common_varieties)<br />

3. Correlation between points and price = 0.4156511636427041<br />
Code :<br />
correlation = df['points'].corr(df['price'])<br />
print("\nCorrelation between points and price:")<br />
print(correlation)<br />

4. Top 5 varieties with the highest average rating<br />
Nebbiolo                     90.251070<br />
Grüner Veltliner             89.980669<br />
Champagne Blend              89.663324<br />
Riesling                     89.450183<br />
Pinot Noir                   89.411468<br />
Code: <br />
top_varieties_rating = df.groupby('variety')['points'].mean().nlargest(5)<br />
print("\ntop 5 varieties with the highest average rating:")<br />
print(top_varieties_rating)<br />

5. Top 10 wineries with the most wine varieties <br />
Testarossa                     217 <br />
Williams Selyem                203<br />
Louis Latour                   199<br />
Georges Duboeuf                195<br />
Chateau Ste. Michelle          187<br />
Wines & Winemakers             181<br />
DFJ Vinhos                     158<br />
Columbia Crest                 144<br />
Concha y Toro                  135<br />
Kendall-Jackson                128<br />
Code :<br />
top_wineries_varieties = df['winery'].value_counts().nlargest(10)<br />
print("\ntop 10 wineries with the most wine varieties:")<br />
print(top_wineries_varieties)<br />

6. Average rating by province<br />
Südburgenland      93.000000<br />
Tokaji             92.000000<br />
Mittelrhein        92.000000<br />
Puente Alto        91.733333<br />
Wachau             91.723724<br />
England            91.581081<br />
Santa Cruz         91.500000<br />
Kamptal            91.466830<br />
Beira Atlantico    91.333333<br />
Traisental         91.255814<br />
Code: <br />

  average_rating_by_province = df.groupby('province')['points'].mean().sort_values(ascending=False).head(10)<br />
  print("\naverage rating by province:")<br />
  print(average_rating_by_province)<br />

