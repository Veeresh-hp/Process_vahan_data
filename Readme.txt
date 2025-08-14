# Vehicle Registration Dashboard

An interactive web dashboard for analyzing vehicle registration data from the Vahan platform, designed with investor insights in mind.

## 🎯 Project Overview

This dashboard provides comprehensive analytics on vehicle registration trends across different categories (2W/3W/4W) and manufacturers, with a focus on Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) growth metrics.

### Key Features

- **Interactive Filtering**: Filter by vehicle type, manufacturer, and time period
- **Growth Analytics**: YoY and QoQ growth visualization and analysis
- **Manufacturer Insights**: Top performer identification and detailed analysis
- **Trend Visualization**: Time series charts showing registration patterns
- **Export Functionality**: Download filtered data for further analysis
- **Responsive Design**: Clean, investor-friendly interface

## 🛠️ Technical Stack

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express & Graph Objects
- **Styling**: Custom CSS for enhanced UI

## 📊 Data Sources

The dashboard expects three CSV files:

1. **vehicle_registrations_cleaned-1.csv**: Raw registration data by month
2. **vehicle_growth_metrics.csv**: Monthly growth calculations
3. **vehicle_growth_metrics_quarterly.csv**: Quarterly aggregated growth metrics

### Data Schema

#### vehicle_registrations_cleaned-1.csv
```
sl.no, Maker, JAN, FEB, MAR, ..., DEC, TOTAL, vehicle_type, year
```

#### vehicle_growth_metrics.csv
```
vehicle_type, Maker, date, year, month, registrations, registrations_last_year, YoY_growth_%, quarter, year_quarter, registrations_last_quarter, QoQ_growth_%
```

#### vehicle_growth_metrics_quarterly.csv
```
vehicle_type, Maker, year, quarter, year_quarter, registrations, registrations_last_quarter, QoQ_growth_%
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd vehicle-registration-dashboard
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
numpy>=1.21.0
```

4. **Place your CSV files**
Ensure your three CSV files are in the project root directory:
- `vehicle_registrations_cleaned-1.csv`
- `vehicle_growth_metrics.csv`
- `vehicle_growth_metrics_quarterly.csv`

5. **Run the application**
```bash
streamlit run vehicle_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 📈 Dashboard Components

### 1. Key Performance Indicators (KPIs)
- Total Registrations
- Average YoY Growth
- Average QoQ Growth
- Active Manufacturers

### 2. Visualization Charts
- **Registration Trends**: Time series showing quarterly registration patterns by vehicle type
- **Top Manufacturers**: Horizontal bar chart of top 10 manufacturers by registrations
- **YoY Growth Analysis**: Line chart showing year-over-year growth trends
- **QoQ Growth Analysis**: Bar chart displaying quarter-over-quarter growth patterns
- **Manufacturer Deep Dive**: Detailed analysis of selected manufacturers

### 3. Interactive Features
- **Multi-select Filters**: Vehicle type, manufacturer, and year selection
- **Dynamic Updates**: All charts update based on filter selections
- **Data Export**: Download filtered data as CSV
- **Responsive Layout**: Optimized for different screen sizes

## 💼 Investor-Focused Insights

The dashboard is specifically designed to provide actionable insights for investors:

1. **Market Trends**: Identify growth patterns across vehicle categories
2. **Manufacturer Performance**: Compare and rank manufacturers by growth metrics
3. **Seasonal Patterns**: Understand quarterly variations in registrations
4. **Growth Momentum**: Track YoY and QoQ growth rates for investment decisions
5. **Market Share Analysis**: Visualize manufacturer market positions

## 🔧 Customization

### Adding New Metrics
To add new metrics, modify the KPI section in the main function:

```python
with col5:  # Add new column
    new_metric = calculate_new_metric(filtered_data)
    st.metric(
        label="New Metric",
        value=f"{new_metric:.1f}",
        delta=f"{new_metric - baseline:.1f}"
    )
```

### Styling Modifications
Update the CSS in the `st.markdown()` section at the top of the file to change colors, fonts, or layout.

### New Chart Types
Add new visualizations using Plotly:

```python
fig_new_chart = px.scatter(
    data,
    x='column1',
    y='column2',
    title="New Chart Title"
)
st.plotly_chart(fig_new_chart, use_container_width=True)
```

## 📱 Deployment Options

### Local Development
```bash
streamlit run vehicle_dashboard.py
```

### Streamlit Community Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with automatic updates

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "vehicle_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure CSV files are in the correct directory
   - Check file names match exactly

2. **Data Type Errors**
   - Verify CSV format matches expected schema
   - Check for missing or malformed data

3. **Memory Issues with Large Datasets**
   - Use data sampling for initial testing
   - Implement pagination for large datasets

4. **Slow Loading**
   - Add `@st.cache_data` decorators to data processing functions
   - Consider data preprocessing and aggregation

## 📊 Performance Optimization

- **Data Caching**: Uses Streamlit's caching for improved performance
- **Selective Loading**: Only processes filtered data for visualizations
- **Efficient Aggregations**: Uses pandas groupby for fast computations
- **Memory Management**: Cleans up intermediate dataframes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Email: [your-email@example.com]

## 🎯 Future Enhancements

- [ ] Real-time data integration with Vahan API
- [ ] Machine learning predictions for future registrations
- [ ] Advanced statistical analysis (correlation, regression)
- [ ] Export to PowerPoint/PDF reports
- [ ] Mobile-responsive design improvements
- [ ] User authentication and personalized dashboards
- [ ] Alert system for significant growth changes
- [ ] Integration with external economic indicators

---

**Built with ❤️ for vehicle industry analysis**