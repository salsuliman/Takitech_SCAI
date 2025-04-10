# Takitech 🏆

<div align="center">
  
**Advanced Football Performance Analysis Platform**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Sportec](https://img.shields.io/badge/Data-Sportec-orange)](https://github.com/kloppy/sportec)
[![Gemini](https://img.shields.io/badge/AI-Gemini-purple)](https://ai.google.dev/gemini-api)

</div>

## 📋 Overview

**Takitech** is a sophisticated football performance analysis platform that leverages advanced data analytics and real-time tactical insights. By processing match and training data, Takitech empowers coaches and analysts to make data-driven decisions that optimize team strategies and player performance.

The platform combines traditional statistical analysis with cutting-edge AI to deliver actionable insights that can transform how teams prepare for and respond during matches.

## ✨ Features

- **Real-time Pressing Intensity Analysis** ⚽ - Track and measure team pressing patterns during live matches
- **LLM-powered Tactical Recommendations** 🧠 - Get AI-generated strategic insights based on performance data
- **Interactive Data Visualization** 📊 - Explore player positions and intensity metrics through dynamic visualizations
- **Video-based Tactical Analysis** 🎥 - Leverage Gemini AI for comprehensive video breakdown and tactical suggestions
- **Custom Performance Metrics** 📈 - Define and track team-specific KPIs that align with your coaching philosophy

## 🔍 Key Functions

### 1. Match Data Processing

```python
match_data = load_match_data(dataset_path, match_id)
```

Collects and processes tracking data from Sportec's open dataset using the `kloppy` library for detailed match analysis with millisecond precision.

### 2. Pressing Intensity Analysis

```python
pressing_data = analyze_pressing_intensity(match_data, team_id, time_window)
```

Analyzes the pressing intensity of players using the **Unravel** library, focusing on key metrics like:
- Ball possession transitions
- Defensive pressure application
- Recovery time after possession loss
- Coordinated pressing movements

### 3. Advanced Data Visualization

```python
create_pressing_heatmap(pressing_data, team_formation)
visualize_player_movements(tracking_data, highlighted_players)
```

Provides sophisticated visualizations using **Matplotlib**, **Seaborn**, and **mplsoccer** libraries:
- Heatmaps showing pressure intensity across the pitch
- Player position tracking with movement vectors
- Pressing intensity time-series visualizations
- Formation analysis diagrams

### 4. Video Creation and Analysis

```python
video_path = create_pressing_intensity_video(model, dataset, OUTPUT_VIDEO)
tactical_analysis = analyze_video_with_gemini(video_path)
```

- Generates high-quality video animations showing pressing intensity analysis over time
- Utilizes `FuncAnimation` for frame-by-frame visual updates
- Uploads video to **Gemini** via the `genai` API
- Generates detailed tactical reports with actionable insights

### 5. Integrated Reporting

```python
generate_performance_report(match_data, pressing_data, tactical_analysis)
```

Creates comprehensive performance reports combining statistical analysis and AI-generated tactical recommendations.

## 🛠️ Technical Architecture

Takitech uses a modular architecture with specialized components:

- **Data Collection Module**: Interfaces with various tracking data providers
- **Analysis Engine**: Processes raw data into tactical insights
- **Visualization Layer**: Transforms analysis into interactive visuals
- **AI Integration**: Connects with Gemini for advanced video analysis
- **Reporting System**: Compiles insights into actionable reports

## 📦 Dependencies

- **Core Data Processing**: `numpy`, `pandas`, `polars`
- **Visualization**: `matplotlib`, `seaborn`, `mplsoccer`
- **Football Analytics**: `kloppy`, `unravel`
- **AI Integration**: `google.genai`
- **Data Source**: `sportec` from Kloppy

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/takitech.git
   cd takitech
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Gemini API key**:
   ```bash
   cp .env.example .env
   # Edit .env file to add your API key
   ```

## 💻 Usage

### Basic Analysis Workflow

```python
from takitech import TakitechAnalyzer

# Initialize the analyzer
analyzer = TakitechAnalyzer(api_key="YOUR_GEMINI_API_KEY")

# Load and analyze match data
match_data = analyzer.load_match_data("path/to/data", match_id=1234)
pressing_analysis = analyzer.analyze_pressing_intensity(
    match_data, 
    team_id="home", 
    time_window=(30, 45)  # 30-45 minute period
)

# Generate visualizations
analyzer.create_pressing_heatmap(pressing_analysis, output_path="pressing_heatmap.png")
video_path = analyzer.create_pressing_intensity_video(pressing_analysis, output_path="pressing_video.mp4")

# Get AI-powered tactical insights
tactical_report = analyzer.analyze_video_with_gemini(video_path)

# Generate comprehensive report
analyzer.generate_report(
    match_data, 
    pressing_analysis, 
    tactical_report, 
    output_path="match_analysis.md"
)
```

### API Key Setup

1. Obtain your **Gemini API Key** from [Google AI Studio](https://ai.google.dev/)
2. Configure it in one of these ways:
   - Set as environment variable: `export GEMINI_API_KEY="your-key-here"`
   - Create a `.env` file with: `GEMINI_API_KEY=your-key-here`
   - Pass directly to the analyzer: `TakitechAnalyzer(api_key="your-key-here")`


### Sample Tactical Analysis
```
The analysis of the pressing pattern reveals a coordinated high-press approach
during minutes 30-45. The team maintains a compact shape with an average 
distance of 7.3m between defensive and midfield lines. 

Key observations:
1. Effective trigger press when opponent CBs receive the ball
2. Right wing showing delayed recovery time (avg +1.2s)
3. Successful turnover rate of 68% in final third pressing
4. Recommendation: Improve coordination between RW and RCM for more effective 
   trap setting on the right flank
```

## 🔄 Integration Options

Takitech can integrate with:
- Video analysis platforms
- Team management systems
- Performance databases
- Custom reporting tools

---

<div align="center">
  <p>Built with ❤️ for football analytics enthusiasts</p>
</div>
