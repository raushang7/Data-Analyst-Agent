# Data Analyst Agent

A powerful AI-powered data analysis API that can source, prepare, analyze, and visualize any data using Large Language Models.

## Features

- **Web Scraping**: Automatically scrapes data from websites, especially Wikipedia tables
- **Multi-format Data Processing**: Supports CSV, Excel, JSON, Parquet, PDF, and text files
- **SQL Query Execution**: Built-in DuckDB support for complex data queries
- **Statistical Analysis**: Correlation analysis, regression, statistical summaries
- **Visualization Generation**: Creates plots, charts, and graphs as base64-encoded images
- **Flexible API**: Accepts questions and data files, returns structured JSON responses

## Project Structure

```
data-analyst-agent/
├── app.py                     # Main Flask API application
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
├── run.sh                    # Local run script
├── README.md                 # This file
└── agent/
    ├── agent.py              # Main agent orchestration logic
    └── tools/
        ├── web_scraper.py    # Web scraping functionality
        ├── data_tools.py     # Data loading and processing
        ├── analysis_tools.py # Statistical analysis and question answering
        └── visualization_tools.py # Chart and plot generation
```

## Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd data-analyst-agent
   ```

2. **Run the setup script**:
   ```bash
   chmod +x run.sh
      ```bash
   PORT=3000 ./run.sh
   ```
   ```

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t data-analyst-agent .
   ```

2. **Run the container**:
   ```bash
   docker run -e PORT=3000 -p 3000:3000 data-analyst-agent
   ```

## API Usage

### Endpoint

- **URL**: `POST /api/`
- **Content-Type**: `multipart/form-data`

### Request Format

Send a POST request with:
- `questions.txt`: Required file containing the analysis questions
- Additional data files (CSV, Excel, JSON, etc.): Optional

### Example Usage

```bash
curl "http://localhost:3000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

### Sample Questions

#### Example 1: Wikipedia Movie Analysis
```text
Scrape the list of highest grossing films from Wikipedia. It is at the URL: https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it. Return as a base-64 encoded data URI under 100,000 bytes.
```

#### Example 2: Court Data Analysis
```text
Answer the following questions and respond with a JSON object containing the answer.
{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}
```

### Response Format

The API returns JSON in one of two formats:

1. **JSON Array** (for numbered questions):
   ```json
   [1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
   ```

2. **JSON Object** (for named questions):
   ```json
   {
     "Which high court disposed the most cases?": "Delhi High Court",
     "What's the regression slope?": 0.123456,
     "Plot visualization": "data:image/png;base64,iVBORw0KG..."
   }
   ```

## Capabilities

### Data Sources
- **Web Scraping**: Wikipedia tables, general websites
- **File Upload**: CSV, Excel, JSON, Parquet, PDF, text files
- **SQL Databases**: DuckDB integration with S3 support

### Analysis Types
- **Counting**: Count records matching specific criteria
- **Comparison**: Find earliest, latest, highest, lowest values
- **Statistics**: Mean, median, correlation, regression analysis
- **Date Analysis**: Calculate delays, time-based patterns

### Visualizations
- **Scatterplots**: With regression lines
- **Time Series**: Year-over-year trends
- **Bar Charts**: Category comparisons
- **Custom Plots**: Based on question requirements

### Special Features
- **Smart Column Detection**: Automatically identifies relevant columns
- **Data Cleaning**: Handles missing values, type conversions
- **Size Optimization**: Compresses images to stay under 100KB limit
- **Error Handling**: Robust error management and logging

## Environment Variables

- `PORT`: Server port (default: 3000)
- `DEBUG`: Enable debug mode (default: False)

## Health Check

Check if the service is running:
```bash
curl http://localhost:3000/health
```

## Development

### Adding New Tools

1. Create a new tool class in `agent/tools/`
2. Add the tool to the agent in `agent/agent.py`
3. Update the analysis plan parser to handle new question types

### Testing

The application includes comprehensive logging and error handling. Check logs for debugging information.

## Deployment Options

### Cloud Platforms
- **Heroku**: Use the included Dockerfile
- **AWS ECS/Fargate**: Container deployment
- **Google Cloud Run**: Serverless container deployment
- **DigitalOcean App Platform**: Simple container deployment

### Local Tunneling
For development testing:
```bash
# Using ngrok
```bash
ngrok http 3000
```

# Using cloudflared
```bash
cloudflared tunnel --url http://localhost:3000
```
```

## Performance Considerations

- **Memory**: Handles large datasets efficiently with pandas and DuckDB
- **Speed**: Optimized for 3-minute response time requirement
- **Concurrency**: Flask with multiple worker processes
- **Caching**: In-memory data processing for speed

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions, please create an issue in the GitHub repository.