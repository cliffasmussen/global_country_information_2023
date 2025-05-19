
# Global Country Information Dataset – Data Visualization Portfolio (2023)

**Author**: Cliff Asmussen 
**Dataset**: world-data-2023.csv  
**Tools**: Python, Pandas, Matplotlib, Seaborn

---

## Project Overview

This project analyzes global country-level data from the year 2023. It uses data visualization to explore and communicate key insights about population, education, military size, and environmental impact.

The goal is to showcase skills in data cleaning, transformation, and visual storytelling. The project focuses on comparisons across countries, continents, and geopolitical groups like the G20.

---

## Visualizations

### 1. Birth Rate vs Gross Tertiary Education Enrollment
- **Purpose**: Explore the relationship between national birth rates and participation in higher education.
- **Visuals**:
  - Global scatter plot: points sized by population and colored by continent.
  - G20-specific version: labeled data points for direct comparison.

### 2. Armed Forces Size vs CO₂ Emissions
- **Purpose**: Investigate whether countries with larger military forces produce more CO₂.
- **Visuals**:
  - Scatter plot with log-log scale.
  - Points sized by population and colored by GDP per capita.

### 3. G20 Countries – Percentage of Population in Armed Forces
- **Purpose**: Compare how militarized G20 countries are in terms of population involvement.
- **Visuals**:
  - Horizontal bar chart showing percent of population in active military service.

---

## Features and Highlights

- Cleaned and standardized data with consistent column naming.
- Custom functions to convert numeric strings (e.g., "1,234") into float format.
- Manually mapped countries to continents for regional analysis.
- Created new features:
  - GDP per capita
  - Average education enrollment rate
  - Percentage of population in armed forces
- Used custom styling and color palettes to enhance visual clarity.
- Logarithmic scales applied to address skewed distributions.

---

## File Structure

```
global_country_insights_2023/
│
├── world-data-2023.csv             # Raw dataset (external source)
├── global_country_insights_2023.py # Main visualization script
└── README.md                       # Project documentation
```

---

## How to Run

1. Ensure the dataset file is downloaded and accessible at the path specified in the script:
   ```python
   df = pd.read_csv(r"C:\path\to\world-data-2023.csv")
   ```

2. Install required Python libraries:
   ```bash
   pip install pandas matplotlib seaborn
   ```

3. Run the script using a Python IDE or in a Jupyter Notebook environment.

---

## Contact

For questions, suggestions, or collaborations, feel free to reach out:

**Cliff Asmussen**  
cliffasmussen@gmail.com
