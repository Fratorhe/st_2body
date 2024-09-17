from io import StringIO

import pandas as pd
import plotly.graph_objects as go


def extract_between_markers(file_path, start_marker, end_marker):
    with open(file_path, 'r') as file:
        content = file.read()

        # Find the start and end of the markers
        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index)

        if start_index == -1 or end_index == -1:
            return "Markers not found"

        # Extract content between markers
        return content[start_index + len(start_marker):end_index].strip()


def reader_ephemeris(filename, columns=None):
    start_marker = '$$SOE'
    end_marker = '$$EOE'
    if columns is None:
        columns = ['JDTDB', 'Calendar Date (TDB)', 'X (km)', 'Y (km)', 'Z (km)', 'VX (km/s)', 'VY (km/s)', 'VZ (km/s)']
    extracted_content = extract_between_markers(filename, start_marker, end_marker)
    # Use StringIO to simulate a file object
    data_io = StringIO(extracted_content)

    # Read the string as a pandas DataFrame
    df = pd.read_csv(data_io, names=columns, index_col=False)
    df = df.drop(columns=['JDTDB', 'Calendar Date (TDB)'])

    return df


def plot_ephemeris(data, label, lift_z=0):
    # the parameter lift_z is used to improve visualizatoin
    plot = go.Scatter3d(
        name=label,
        x=data['X (km)'],
        y=data['Y (km)'],
        z=data['Z (km)']+lift_z,
        mode='markers',
        marker=dict(
            size=2,
            # color=data.index,
            # colorscale='Cividis',  # Different colorscale for contrast
            opacity=0.2
        )
    )
    return plot

def create_figure(list_of_plots, axis_range=(-6e8,6e8)):
    # Create the figure and add the plots
    fig = go.Figure(data=list_of_plots)

    # Update layout to set the same range for all three axes
    if axis_range:
        fig.update_layout(
            scene=dict(
            xaxis=dict(range=axis_range, title='X Axis'),
            yaxis=dict(range=axis_range, title='Y Axis'),
            zaxis=dict(range=axis_range, title='Z Axis')
        ),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(
                font=dict(size=18),  # Change the font size here
                itemsizing='constant',  # This helps to ensure items are sized properly
                itemwidth=40,  # Adjust item width
            ),
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(
                font=dict(size=18),  # Change the font size here
                itemsizing='constant',  # This helps to ensure items are sized properly
                itemwidth=40,  # Adjust item width
            ),
        )
    return fig