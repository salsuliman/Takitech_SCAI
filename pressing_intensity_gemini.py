import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from matplotlib.animation import FuncAnimation
from mplsoccer import VerticalPitch
import matplotlib.patheffects as path_effects
from google import genai
from kloppy import sportec
from unravel.soccer import KloppyPolarsDataset, PressingIntensity

# Constants for visualization
HOME_COLOR, HOME_GK_COLOR = "red", "grey"
AWAY_COLOR, AWAY_GK_COLOR = "black", "green"
BALL_COLOR = "orange"
OUTPUT_VIDEO = "pressing_intensity_analysis.mp4"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your actual API key

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def load_match_data(match_id, limit=5000):
    """Load tracking data for a match from Sportec's open dataset."""
    print(f"Loading tracking data for match {match_id}...")
    coordinates = "secondspectrum"
    kloppy_dataset = sportec.load_open_tracking_data(
        match_id=match_id, coordinates=coordinates, limit=limit
    )
    return KloppyPolarsDataset(kloppy_dataset=kloppy_dataset, orient_ball_owning=False)

def analyze_pressing_intensity(dataset, start_time, end_time, period_id=1):
    """Analyze pressing intensity using the Unravel library's model."""
    print("Analyzing pressing intensity...")
    model = PressingIntensity(dataset=dataset)
    
    model.fit(
        start_time=start_time,
        end_time=end_time,
        period_id=period_id,
        method="teams",
        ball_method="max",
        orient="home_away",
        speed_threshold=2.0,
    )
    
    return model

def plot_settings(ax, row_players, column_players, speed_threshold=None):
    """Configure plot settings for the heatmap."""
    for t in ax.texts:
        t.set_text(t.get_text() + " %")
    
    if hasattr(ax.figure, 'axes') and len(ax.figure.axes) > 1:
        ax.figure.axes[-1].yaxis.label.set_size(10)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
        length=0
    )
    
    ax.xaxis.set_label_position("top")

    # Set colors for player labels
    row_colors = [
        (
            HOME_COLOR if player is not None and player.is_home and not player.is_gk
            else HOME_GK_COLOR if player is not None and player.is_home and player.is_gk
            else AWAY_COLOR if player is not None and not player.is_home and not player.is_gk
            else AWAY_GK_COLOR if player is not None and not player.is_home and player.is_gk
            else BALL_COLOR
        )
        for player in row_players
    ]
    
    column_colors = [
        (
            HOME_COLOR if player is not None and player.is_home and not player.is_gk
            else HOME_GK_COLOR if player is not None and player.is_home and player.is_gk
            else AWAY_COLOR if player is not None and not player.is_home and not player.is_gk
            else AWAY_GK_COLOR if player is not None and not player.is_home and player.is_gk
            else BALL_COLOR
        )
        for player in column_players
    ]

    # Apply colors to tick labels
    for t, color in zip(ax.xaxis.get_ticklabels(), column_colors):
        t.set_color(color)
    
    for t, color in zip(ax.yaxis.get_ticklabels(), row_colors):
        t.set_color(color)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    # Set team names as labels if using team method
    fontsize = 15
    if len(row_players) > 0 and len(column_players) > 0:
        if hasattr(row_players[0], 'team_name') and hasattr(column_players[0], 'team_name'):
            ax.set_ylabel(row_players[0].team_name, fontsize=fontsize)
            ax.set_xlabel(column_players[0].team_name, fontsize=fontsize)
        else:
            ax.set_ylabel("", fontsize=fontsize)
            ax.set_xlabel("", fontsize=fontsize)

    # Remove percentage sign from text
    for t in ax.texts:
        t.set_text(t.get_text().replace(" %", ""))
    
    if speed_threshold is not None:
        ax.set_title(f"Active Pressing [v > {speed_threshold}m/s]", fontsize=14)

def plot_player_positions(frame_data, ax, dataset):
    """Plot player and ball positions on the pitch."""
    for r in frame_data.iter_rows(named=True):
        v, vy, vx, y, x = r["v"], r["vx"], r["vy"], r["x"], r["y"]
        is_ball = True if r["team_id"] == "ball" else False

        if not is_ball:
            player = dataset.get_player_by_id(player_id=r["id"])
            
            color = (
                HOME_COLOR if player.is_home and not player.is_gk
                else HOME_GK_COLOR if player.is_home
                else AWAY_COLOR if not player.is_gk 
                else AWAY_GK_COLOR
            )
            
            ax.scatter(x, y, color=color, s=150)

            # Draw velocity vector for players moving faster than 1 m/s
            if v > 1.0:
                ax.annotate(
                    "",
                    xy=(x + vx, y + vy),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=3),
                )
            
            # Add player number with white border
            text = ax.text(
                x, y, player.number,
                color=color, fontsize=8,
                ha="center", va="center", zorder=5
            )
            
            text.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground="white"),
                path_effects.Normal()
            ])
        else:
            # Draw the ball
            ax.scatter(x, y, color=BALL_COLOR, s=50, zorder=10)

def plot_intensity_matrix(matrix, row_players, column_players, ax, speed_threshold=None):
    """Plot the pressing intensity heatmap."""
    df = pd.DataFrame(
        data=matrix,
        index=[p.number if p is not None else "ball" for p in row_players],
        columns=[p.number if p is not None else "ball" for p in column_players],
    )
    
    sns.heatmap(
        df * 100,  # Convert to percentage
        xticklabels=True,
        yticklabels=True,
        cmap="hot_r",
        ax=ax,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        square=True,
        linewidths=0.5,
        cbar=False,
    )
    
    plot_settings(ax, row_players, column_players, speed_threshold)
    return ax

def create_pressing_intensity_video(model, dataset, output_path):
    """Create a video showing pressing intensity analysis over time."""
    print(f"Creating pressing intensity video: {output_path}")
    
    # Setup the pitch visualization
    coordinates = "secondspectrum"
    pitch = VerticalPitch(
        pitch_type=coordinates,
        pitch_length=dataset.settings.pitch_dimensions.pitch_length,
        pitch_width=dataset.settings.pitch_dimensions.pitch_width,
        pitch_color="white",
        line_color="#343131",
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    def update(idx):
        """Update function for animation."""
        ax1.clear()
        ax2.clear()

        # Draw the pitch
        pitch.draw(ax=ax1)
        
        # Get data for the current frame
        row = model.output.to_pandas().iloc[idx]
        
        # Get player IDs for the current frame
        row_players = [dataset.get_player_by_id(player_id) for player_id in row["rows"]]
        column_players = [dataset.get_player_by_id(player_id) for player_id in row["columns"]]
        
        # Get frame data
        frame_data = dataset.filter(
            (pl.col("frame_id") == row["frame_id"]) & 
            (pl.col("period_id") == row["period_id"])
        )
        
        # Plot player positions
        plot_player_positions(frame_data=frame_data, ax=ax1, dataset=dataset)
        
        # Plot intensity matrix
        plot_intensity_matrix(
            matrix=np.array([x for x in row["probability_to_intercept"]]),
            row_players=row_players,
            column_players=column_players,
            speed_threshold=model._speed_threshold,
            ax=ax2,
        )

    # Create animation
    frame_count = min(len(model.output), 500)  # Limit frames for reasonable file size
    ani = FuncAnimation(fig, update, frames=range(frame_count), repeat=False)
    
    # Save animation
    try:
        frame_rate = dataset.kloppy_dataset.metadata.frame_rate
    except AttributeError:
        frame_rate = 25  # Default frame rate if not available
        
    ani.save(
        output_path, 
        fps=frame_rate, 
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
    )
    
    plt.close(fig)
    return output_path

def analyze_video_with_gemini(video_path):
    """Upload the video to Gemini and get tactical analysis."""
    print(f"Uploading video to Gemini for analysis: {video_path}")
    
    # Upload the video file
    video_file = client.files.upload(file=video_path)
    print(f"Completed upload: {video_file.uri}")
    
    # Check if video processing is complete
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    
    print('Video processing complete. Generating analysis...')
    
    # Generate tactical analysis
    prompt = """
    Analyze this football pressing intensity visualization in detail:
    1. Identify patterns in pressing intensity between teams
    2. Highlight individual players who are most effective at pressing
    3. Suggest tactical adjustments for both teams to improve their pressing
    4. Provide specific insights that a coach could use for half-time team talk
    5. Recommend 3 specific drills to improve pressing efficiency
    
    Format your analysis as a comprehensive tactical report.
    """
    
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=[video_file, prompt]
    )
    
    return response.text

def main():
    """Main function to run the entire analysis pipeline."""
    # 1. Load match data
    match_id = "J03WMX"  # Replace with actual match ID
    dataset = load_match_data(match_id)
    
    # 2. Define time period to analyze
    start_time = pl.duration(minutes=1, seconds=53)
    end_time = pl.duration(minutes=2, seconds=32)
    period_id = 1
    
    # 3. Analyze pressing intensity
    model = analyze_pressing_intensity(dataset, start_time, end_time, period_id)
    
    # 4. Create video visualization
    video_path = create_pressing_intensity_video(model, dataset, OUTPUT_VIDEO)
    
    # 5. Get Gemini analysis of the video
    tactical_analysis = analyze_video_with_gemini(video_path)
    
    # 6. Save tactical analysis to file
    with open("tactical_analysis.md", "w") as f:
        f.write(tactical_analysis)
    
    print(f"Analysis complete. Tactical report saved to tactical_analysis.md")
    
    return tactical_analysis

if __name__ == "__main__":
    main()