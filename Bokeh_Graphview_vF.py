# Import libararies
import pandas as pd
import numpy as np

from bokeh.layouts import row, widgetbox, column
from bokeh.models import ColumnDataSource, StaticLayoutProvider, Circle
from bokeh.models import HoverTool, BoxSelectTool, GraphRenderer, NumberFormatter
from bokeh.models.widgets import RangeSlider, DataTable, TableColumn, NumberFormatter, Div
from bokeh.io import curdoc, show, output_notebook
from bokeh.plotting import figure

import networkx as nx

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes, NodesOnly

# Import / instantiate networkx graph
edgelist_df = pd.read_csv('projected_graph_edgelist.csv')

edgelist_df = edgelist_df[['group1','group2','bayes']]
edgelist_df.columns = ['source', 'target', 'weight']

# Create NetworkX graph from edges
G = nx.from_pandas_edgelist(edgelist_df,edge_attr=True)

# Get node_source information from node list 
nodelist_df = pd.read_csv('projected_graph_nodelist.csv')
node_source = ColumnDataSource(nodelist_df)

node_size = list(nodelist_df['node_size'])
node_color = list(nodelist_df['node_color'])

# Import positions to ensure the nodes stay static with changes to the sliders and to improve speed
positions = np.load('position_weights.npy').item()

# Initialize the first graph
G_source = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))
graph = GraphRenderer()

# Initialize the data_table_source and summary_data_table_source
data_table_source = ColumnDataSource(data=dict())
summary_data_table_source = ColumnDataSource(data=dict())

# Update loop which evaluates which edges are within slider threshold and renders changes
def update():
    selected_df = edgelist_df[(edgelist_df['weight'] >= slider.value[0]) & (edgelist_df['weight'] <= slider.value[1])]
    sub_G = nx.from_pandas_edgelist(selected_df,edge_attr=True)
    sub_graph = from_networkx(sub_G, nx.spring_layout, scale=2, center=(0,0))
    graph.edge_renderer.data_source.data = sub_graph.edge_renderer.data_source.data
    graph.node_renderer.data_source.data = G_source.node_renderer.data_source.data
    graph.node_renderer.data_source.add(node_size,'node_size')
    graph.node_renderer.data_source.add(node_color,'node_color')
    
# Updates tables to provide summary statistics for selected nodes
def selected_points(attr,old,new):
    selected_idx = graph.node_renderer.data_source.selected.indices
    data_table_source.data = {'group_name' : list(np.array(node_source.data['group_name'])[selected_idx]),
                              'member_count' : list(np.array(node_source.data['member_count'])[selected_idx]),
                              'degree_centrality' : list(np.array(node_source.data['degree_centrality'])[selected_idx]),
                              'rel_closeness_centrality' : list(np.array(node_source.data['rel_closeness_centrality'])[selected_idx])}
    summary_data_table_source.data = {'summary_stats': list(['Mean Values']),
                          'mean_member_count' : [np.mean(np.array(node_source.data['member_count'])[selected_idx])],
                          'mean_degree_centrality' : [np.mean(np.array(node_source.data['degree_centrality'])[selected_idx])],
                          'mean_rel_closeness_centrality' : [np.mean(np.array(node_source.data['rel_closeness_centrality'])[selected_idx])]}

# Slider parameters which changes values to update the graph
slider = RangeSlider(title="Weights", start=0, end=1, value=(0.25, 0.75), step=0.10)
slider.on_change('value', lambda attr, old, new: update())

# Plot object which is updated 
plot = figure(title="Meetup Network Analysis", x_range=(-1.4,2.6), y_range=(-2.0,2.0),
             tools = "pan,wheel_zoom,box_select,reset,box_zoom,crosshair", plot_width=800, plot_height=700)

# Assign layout for nodes, render graph, and add hover tool
graph.node_renderer.data_source.selected.on_change("indices", selected_points)
graph.layout_provider = StaticLayoutProvider(graph_layout=positions)
graph.node_renderer.glyph = Circle(size='node_size', fill_color='node_color')
graph.selection_policy = NodesOnly()
plot.renderers.append(graph)
plot.tools.append(HoverTool(tooltips=[('Name', '@index')]))

# Create Summary Data Table
num_format = NumberFormatter(format="0.00")
summary_table_title = Div(text="""<b>Summary Statistics</b>""", width=525, height=10)
summary_table_cols = [TableColumn(field='summary_stats', title="SummaryStats"),
                      TableColumn(field='mean_member_count', title="Member Count",formatter=num_format),
                      TableColumn(field='mean_degree_centrality', title="Degree Centrality",formatter=num_format),
                      TableColumn(field='mean_rel_closeness_centrality', title="Rel. Closeness Centrality",formatter=num_format)]
summary_data_table = DataTable(source=summary_data_table_source,
                               columns=summary_table_cols, width=525, height=80)

# Create Data Table
data_table_cols = [TableColumn(field="group_name", title="Node Name"),
                   TableColumn(field="member_count", title="Member Count",formatter=num_format),
                   TableColumn(field="degree_centrality", title="Degree Centrality",formatter=num_format),
                   TableColumn(field="rel_closeness_centrality", title="Rel. Closeness Centrality",formatter=num_format)]

data_table = DataTable(source=data_table_source, columns=data_table_cols, width=525, height=550)
data_table_title = Div(text="""<b>Selected Data List</b>""", width=525, height=10)

# Set layout of objects
layout = row(column(slider,plot),
             column(summary_table_title,summary_data_table,data_table_title,data_table))

# Create Bokeh server object to render changes 
curdoc().add_root(layout)
update()