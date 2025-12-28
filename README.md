Data Cleaning & Preparation Issues Encountered

Inconsistent missing value representations
Missing values were encoded using multiple symbols and strings ('-', '—', empty strings, 'None', 'null',), which were not detected by isnull() until standardized.

Numeric data stored as text with units
Several numeric columns contained units or text:

Engine volume included values such as 2.0 Turbo, 1.8T

mileage contained values like 120000 km
These required text parsing and conversion to numeric types.

Mixed semantic information in a single column
The Engine volume column combined:

numeric engine size

turbo information
This required feature splitting into:

engine_volume (numeric)

is_turbo (binary indicator)

Implicit categorical data misinterpreted as datetime
The doors column was auto-parsed as a date due to values such as 02-Mar, requiring explicit casting to string to preserve categorical meaning.

Inconsistent categorical encodings
Categorical columns such as drive_wheels contained heterogeneous labels (front, rear, 4x4) that needed consistent categorical handling rather than numeric conversion.

Column name inconsistencies causing pipeline failures
Errors occurred due to:

mismatched capitalization

trailing spaces

renamed or normalized column headers
This caused KeyError failures inside ColumnTransformer.

False assumption of missing columns due to display truncation
Pandas output truncated wide DataFrames, making columns appear “missing” when they were only hidden from display.

Index confusion after filtering operations
Filtered DataFrame views retained original row indices, which appeared non-sequential and were initially mistaken for indexing errors.

Target leakage risk avoided by correct preprocessing order
Encoding and scaling were properly placed inside the pipeline after train–test splitting, preventing leakage.




