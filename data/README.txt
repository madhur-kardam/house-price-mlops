# data/ folder

Place your `housing.csv` dataset file here.

## How to get the dataset:

1. Go to: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
2. Click Download (free Kaggle account required)
3. Unzip the file
4. Rename the CSV to `housing.csv`
5. Place it in this `data/` folder

## Expected columns:

| Column     | Type  | Description              |
|------------|-------|--------------------------|
| price      | int   | House price (target)     |
| area       | int   | Area in square feet      |
| bedrooms   | int   | Number of bedrooms       |
| bathrooms  | int   | Number of bathrooms      |
| stories    | int   | Number of floors         |
| parking    | int   | Number of parking spots  |

The dataset also has other columns (mainroad, guestroom, etc.)
that we ignore — train.py only uses the 5 columns listed above.
